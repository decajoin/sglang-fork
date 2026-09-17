# SPDX-License-Identifier: Apache-2.0
"""Streaming long-video stage: one request generated as a chain of chunks.

A MiniMax H3 request is capped at 4-15 seconds. This stage serves longer
requests by running latent preparation and denoising once per fixed-length
chunk, with every chunk after the first reading the earlier chunks' clean
attention K/V as history. Nothing is decoded and re-encoded between chunks, so
there is no per-hop round trip whose gain could compound over a long chain.

A continuation also pins its opening frame with an fl2va ``frame_index=0``
condition, but the rows come from the previous chunk's own last single-frame
latent rather than a decoded frame. That is the model's own first-frame
conditioning path, and it costs no VAE round trip.

Because the past arrives as K/V, a continuation omits the five-frame affine
prefix that opens a standalone request and generates only the steady 17-frame
groups: 124 frames for the first chunk, 119 for each continuation, nothing
reproduced and nothing dropped at publication.

The joined latents are decoded in windows rather than all at once. The decoder
tiles only height and width, so a one-shot decode's memory grows with the whole
clip; windows keep it flat. See MINIMAX_H3_DECODE_WINDOW_LATENTS.

Requests without ``total_duration_seconds`` take the single-shot path, which is
the unmodified four-stage sequence this stage replaces.
"""
from __future__ import annotations

from typing import Any

import msgspec
import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


# Per-chunk audio rounding leaves the joined track a fraction of a second off
# the video. More than this is a geometry error, not rounding.
MINIMAX_H3_AV_JOIN_TOLERANCE_S = 0.25

# The decoder's temporal cost is linear in the whole clip -- klvae tiles only
# height and width, so every spatial tile carries the full T. Decoding the join
# in windows keeps that cost flat at one window instead of growing with the
# request. Measured at 768x1344: 67.3 MiB per raw frame, so a 12-chunk clip
# costs 110.9 GiB one-shot against 9.8 GiB per window.
#
# A window may only start where the decoder's own (1,4,4,4,4) frame weighting
# is in phase, which is a latent index divisible by 5, and its length must
# satisfy the 5n+2 contract that frame_count_from_video_latent_t enforces.
# 37 latents starting at 35k satisfies both, advances exactly one chunk, and
# overlaps its predecessor by the 5 frames a continuation already discards.
MINIMAX_H3_DECODE_WINDOW_LATENTS = 37
MINIMAX_H3_DECODE_WINDOW_STRIDE = 35
MINIMAX_H3_DECODE_WINDOW_OVERLAP_FRAMES = 5
# Audio is decoded once, with the first window. Later windows still run the
# audio VAE because the decode stage does both, so they get a stub whose only
# job is to be cheap; its output is discarded.
MINIMAX_H3_DECODE_AUDIO_STUB_LATENTS = 8


class MiniMaxH3StreamingChunkStage(PipelineStage):
    """Run latent-prep -> denoise -> decode once per chunk and join the results."""

    def __init__(
        self,
        *,
        text_encoding,
        latent_preparation,
        timestep_preparation,
        denoising,
        decoding,
        video_vae,
        vae_arch_config,
    ) -> None:
        super().__init__()
        self._text_encoding = text_encoding
        self._latent_preparation = latent_preparation
        self._timestep_preparation = timestep_preparation
        self._denoising = denoising
        self._decoding = decoding
        self.video_vae = video_vae
        self.vae_arch_config = vae_arch_config

    # ---- sub-stage plumbing (mirrors ProgressiveDenoisingStageRouter) ----

    @property
    def _sub_stages(self) -> tuple[PipelineStage, ...]:
        return (
            self._text_encoding,
            self._latent_preparation,
            self._timestep_preparation,
            self._denoising,
            self._decoding,
        )

    def set_component_residency_manager(self, manager) -> None:
        super().set_component_residency_manager(manager)
        for stage in self._sub_stages:
            stage.set_component_residency_manager(manager)

    def set_registered_stage_name(self, stage_name: str) -> None:
        super().set_registered_stage_name(stage_name)
        for stage in self._sub_stages:
            stage.set_registered_stage_name(stage_name)

    def set_profile_stage_name(self, stage_name: str) -> None:
        super().set_profile_stage_name(stage_name)
        for stage in self._sub_stages:
            stage.set_profile_stage_name(stage_name)

    @property
    def role_affinity(self) -> RoleType:
        # The chunk loop ends in decode, so this stage owns the decoder role.
        return RoleType.DECODER

    @property
    def parallelism_type(self) -> StageParallelismType:
        return self._decoding.parallelism_type

    def component_uses(self, server_args: ServerArgs, stage_name: str | None = None):
        """Declare every component the chunk loop touches, in call order.

        The residency manager matches an actual use against this list and
        tolerates repeats by rescanning from the start, so one pass of the
        loop body is the right declaration for an N-chunk request.
        """
        stage_name = self._component_stage_name(stage_name)
        uses: list[Any] = []
        seen: set[tuple[str, Any]] = set()
        # Declaration order follows the call order: chunk prompts are encoded
        # first, then every chunk runs the DiT and both VAEs.
        for stage in (self._text_encoding, self._denoising, self._decoding):
            for use in stage.component_uses(server_args, stage_name):
                key = (use.component_name, use.phase)
                if key in seen:
                    continue
                seen.add(key)
                uses.append(use)
        return uses

    # ---- entry point ----

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        total_duration = getattr(
            batch.sampling_params, "total_duration_seconds", None
        )
        if total_duration is None:
            return self._run_single_shot(batch, server_args)
        return self._run_streaming(batch, server_args, float(total_duration))

    def _run_single_shot(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        """The unmodified four-stage sequence, for ordinary 4-15s requests."""
        self._latent_preparation.forward(batch, server_args)
        self._timestep_preparation.forward(batch, server_args)
        self._denoising.forward(batch, server_args)
        return self._decoding.forward(batch, server_args)

    # ---- streaming path ----

    def _run_streaming(
        self,
        batch: Req,
        server_args: ServerArgs,
        total_duration_seconds: float,
    ) -> OutputBatch:
        """Generate the clip as a chain of chunks sharing one clean K/V cache.

        No chunk is decoded on its own. A continuation omits the five-frame
        affine prefix that opens a standalone request, so its latents only
        describe 119 frames as part of the whole clip; decoded alone the VAE
        would read the same latents as a 107-frame clip with a prefix of its
        own. The chain therefore accumulates latents and decodes once.
        """
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
            MINIMAX_H3_STREAMING_CHUNK_EXTRA_KEY,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            minimax_h3_plan_from_batch,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.streaming_kv import (
            MiniMaxH3StreamingChunkContext,
            MiniMaxH3StreamingKVCache,
            minimax_h3_streaming_anchor_rows,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
            minimax_h3_streaming_chunk_plan,
        )

        plan = minimax_h3_plan_from_batch(batch)
        if plan is None:
            raise ValueError(
                "streaming long video requires a canonical MiniMax H3 request"
            )
        chunk_plan = minimax_h3_streaming_chunk_plan(
            total_duration_seconds=total_duration_seconds,
            chunk_seconds=float(server_args.streaming_chunk_seconds),
        )
        prompts = self._resolve_chunk_prompts(batch, plan, chunk_plan.chunk_count)
        base_seed = 42 if plan.seed is None else int(plan.seed)

        self._timestep_preparation.forward(batch, server_args)
        embeddings = self._precompute_text_embeddings(
            batch, server_args, plan, prompts
        )

        cache = MiniMaxH3StreamingKVCache()
        recent = getattr(batch.sampling_params, "streaming_kv_recent_chunks", None)
        video_only_sink = getattr(
            batch.sampling_params, "streaming_kv_video_only_sink", None
        )
        stream = MiniMaxH3StreamingChunkContext(
            cache=cache,
            chunk_index=0,
            video_latent_index_origin=0,
            audio_latent_index_origin=0,
            recent_chunks=1 if recent is None else int(recent),
            video_only_sink=True if video_only_sink is None else bool(video_only_sink),
        )
        anchor: dict[str, Any] | None = None
        video_latents: list[torch.Tensor] = []
        audio_latents: list[torch.Tensor] = []
        try:
            for index in range(chunk_plan.chunk_count):
                stream.chunk_index = index
                stream.video_latent_index_origin = (
                    chunk_plan.video_latent_offset_for_chunk(index)
                )
                stream.audio_latent_index_origin = (
                    chunk_plan.audio_latent_offset_for_chunk(index)
                )
                batch.extra[MINIMAX_H3_STREAMING_CHUNK_EXTRA_KEY] = stream
                self._install_chunk_state(
                    batch,
                    plan=plan,
                    chunk_plan=chunk_plan,
                    chunk_index=index,
                    seed=base_seed + index,
                    embeddings=embeddings[index],
                    anchor=anchor,
                )
                self._latent_preparation.forward(batch, server_args)
                self._denoising.forward(batch, server_args)
                video_latents.append(batch.latents)
                audio_latents.append(batch.audio_latents)
                if index + 1 < chunk_plan.chunk_count:
                    anchor = self._latent_anchor_payload(
                        batch.latents,
                        latent_index_origin=stream.video_latent_index_origin,
                        frame_count=chunk_plan.frames_for_chunk(index + 1),
                        rows=minimax_h3_streaming_anchor_rows(
                            batch.latents,
                            latent_index_origin=stream.video_latent_index_origin,
                        ),
                    )
        finally:
            batch.extra.pop(MINIMAX_H3_STREAMING_CHUNK_EXTRA_KEY, None)
            cache.clear()

        per_chunk_audio = [int(part.shape[-1]) for part in audio_latents]
        expected_audio = [
            chunk_plan.audio_latent_t_for_chunk(index)
            for index in range(chunk_plan.chunk_count)
        ]
        if per_chunk_audio != expected_audio:
            raise RuntimeError(
                f"streaming chunks produced {per_chunk_audio} audio latents, "
                f"plan expected {expected_audio}"
            )
        per_chunk = [int(part.shape[2]) for part in video_latents]
        expected = [
            chunk_plan.video_latent_t_for_chunk(index)
            for index in range(chunk_plan.chunk_count)
        ]
        if per_chunk != expected:
            # The joined length is what the one-shot decode turns into frames,
            # so a chunk that produced the wrong latent count has to say so
            # here rather than surface as an off-by-a-few frame count.
            raise RuntimeError(
                f"streaming chunks produced {per_chunk} video latents, "
                f"plan expected {expected}"
            )
        batch.latents = torch.cat(video_latents, dim=2)
        batch.audio_latents = torch.cat(audio_latents, dim=-1)
        video_latents.clear()
        audio_latents.clear()
        self._restore_published_plan(batch, plan, chunk_plan)
        output = self._decode_windowed(batch, server_args, plan, chunk_plan)
        published = int(output.output.shape[2])
        if published != chunk_plan.published_frames:
            raise RuntimeError(
                f"streaming decode produced {published} frames, plan expected "
                f"{chunk_plan.published_frames}"
            )
        sample_rate = int(output.audio_sample_rate)
        video_seconds = published / chunk_plan.fps
        audio_seconds = int(output.audio.shape[-1]) / sample_rate
        if abs(audio_seconds - video_seconds) > MINIMAX_H3_AV_JOIN_TOLERANCE_S:
            raise RuntimeError(
                f"streaming decode joined {audio_seconds:.3f}s of audio against "
                f"{video_seconds:.3f}s of video, beyond the "
                f"{MINIMAX_H3_AV_JOIN_TOLERANCE_S:g}s join tolerance"
            )
        # Settle the residue against the frames actually published. The muxer
        # stretches the shorter stream to the longer one, so an audio track even
        # a fraction of a second long silently changes the delivered frame count.
        output.audio = self._fit_audio(
            output.audio,
            published_frames=published,
            fps=chunk_plan.fps,
            sample_rate=sample_rate,
        )
        self._publish_resolved_length(batch, published_frames=published)
        return output

    def _decode_windowed(
        self, batch: Req, server_args: ServerArgs, plan, chunk_plan
    ) -> OutputBatch:
        """Decode the joined latents in windows instead of all at once.

        The windows tile the clip exactly: window ``k`` covers latents
        ``[35k, 35k+37)``, which is frames ``[119k, 119k+124)``. Dropping the
        leading 5 frames of every window but the first leaves the same
        partition the chunks themselves published, and the last window lands on
        the final latent with nothing left over.

        The decode stage decodes video and audio from independent tensors, so
        only the video is windowed; the first window carries the whole audio
        track and later windows carry a stub whose output is thrown away.
        """
        video_latents = batch.latents
        audio_latents = batch.audio_latents
        total_latents = int(video_latents.shape[2])
        window, stride = (
            MINIMAX_H3_DECODE_WINDOW_LATENTS,
            MINIMAX_H3_DECODE_WINDOW_STRIDE,
        )
        starts = list(range(0, total_latents - window + 1, stride))
        if not starts or starts[-1] + window != total_latents:
            raise RuntimeError(
                f"{total_latents} joined latents do not tile into {window}-latent "
                f"windows at stride {stride}; the join geometry changed"
            )
        if len(starts) != chunk_plan.chunk_count:
            raise RuntimeError(
                f"decode windows ({len(starts)}) disagree with the chunk count "
                f"({chunk_plan.chunk_count})"
            )

        audio_stub = audio_latents[..., :MINIMAX_H3_DECODE_AUDIO_STUB_LATENTS]
        window_frames = chunk_plan.frames_per_chunk
        video_parts: list[torch.Tensor] = []
        audio: torch.Tensor | None = None
        sample_rate: int | None = None
        last_output: OutputBatch | None = None
        try:
            for index, start in enumerate(starts):
                if start % 5:
                    raise RuntimeError(
                        f"decode window {index} starts at latent {start}, which is "
                        "out of phase with the (1,4,4,4,4) frame weighting"
                    )
                first = index == 0
                batch.latents = video_latents[:, :, start : start + window]
                batch.audio_latents = audio_latents if first else audio_stub
                self._install_decode_window_plan(
                    batch, plan, chunk_plan, frame_count=window_frames
                )
                output = self._decoding.forward(batch, server_args)
                last_output = output
                frames = output.output
                if int(frames.shape[2]) != window_frames:
                    raise RuntimeError(
                        f"decode window {index} produced {int(frames.shape[2])} "
                        f"frames, expected {window_frames}"
                    )
                drop = 0 if first else MINIMAX_H3_DECODE_WINDOW_OVERLAP_FRAMES
                video_parts.append(self._to_host(frames[:, :, drop:]))
                if first:
                    audio = self._to_host(output.audio)
                    sample_rate = int(output.audio_sample_rate)
                del output, frames
        finally:
            batch.latents = video_latents
            batch.audio_latents = audio_latents
            self._restore_published_plan(batch, plan, chunk_plan)

        if last_output is None or audio is None or sample_rate is None:
            raise RuntimeError("windowed decode produced no output")
        last_output.output = torch.cat(video_parts, dim=2)
        last_output.audio = audio
        last_output.audio_sample_rate = sample_rate
        return last_output

    def _install_decode_window_plan(
        self, batch: Req, plan, chunk_plan, *, frame_count: int
    ) -> None:
        """Point the resolved plan at the window the decoder is about to see."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY,
        )

        shape = dict(plan.shape)
        shape["frame_count"] = int(frame_count)
        shape["video_latent_t"] = int(batch.latents.shape[2])
        shape["audio_latent_t"] = int(batch.audio_latents.shape[-1])
        batch.extra[MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY] = msgspec.structs.replace(
            plan, task="t2va", shape=shape
        )
        batch.raw_latent_shape = tuple(batch.latents.shape)

    def _install_chunk_state(
        self,
        batch: Req,
        *,
        plan,
        chunk_plan,
        chunk_index: int,
        seed: int,
        embeddings: Any,
        anchor: dict[str, Any] | None = None,
    ) -> None:
        """Rewrite the per-chunk extras for a cross-chunk-KV chunk.

        The chunk's own geometry is written into the plan shape rather than
        derived from its duration: a continuation is not a standalone request,
        and the 17n+5 frame formula would give it the wrong latent count.
        """
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
            MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
            MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY,
            MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY,
        )

        batch.extra.pop(MINIMAX_H3_DENOISE_STATE_EXTRA_KEY, None)
        batch.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY] = embeddings
        if anchor is None:
            batch.extra.pop(MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY, None)
        else:
            batch.extra[MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY] = anchor

        shape = dict(plan.shape)
        shape["frame_count"] = chunk_plan.frames_for_chunk(chunk_index)
        shape["video_latent_t"] = chunk_plan.video_latent_t_for_chunk(chunk_index)
        shape["audio_latent_t"] = chunk_plan.audio_latent_t_for_chunk(chunk_index)
        task = "t2va" if anchor is None else "fl2va"
        batch.extra[MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY] = msgspec.structs.replace(
            plan, seed=seed, task=task, shape=shape
        )
        self._sync_canonical_task(batch, task)

    @staticmethod
    def _latent_anchor_payload(
        latents: torch.Tensor,
        *,
        latent_index_origin: int,
        frame_count: int,
        rows: torch.Tensor,
    ) -> dict[str, Any]:
        """Wrap anchor rows as the fl2va first-frame condition payload."""
        latent_h = int(latents.shape[3])
        latent_w = int(latents.shape[4])
        entry = {
            "rows": rows,
            "latent_h": latent_h,
            "latent_w": latent_w,
            "canvas_height": latent_h * 16,
            "canvas_width": latent_w * 16,
            "frame_index": 0,
            "resolved_frame_index": 0,
            "condition_index": 0,
        }
        return {
            "rows": rows,
            "latent_h": latent_h,
            "latent_w": latent_w,
            "canvas_height": latent_h * 16,
            "canvas_width": latent_w * 16,
            "keyframes": [entry],
            "semantic_frame_indices": (0,),
            "pixel_frame_indices": [0],
            "frame_count": int(frame_count),
        }

    @classmethod
    def _restore_published_plan(cls, batch: Req, plan, chunk_plan) -> None:
        """Describe the joined clip before it is decoded.

        The last chunk left the plan describing itself -- under kv_anchor an
        fl2va chunk with its own anchor -- and the decoder is about to see
        every chunk at once as one plain clip. The canonical request has to
        follow the plan's task here, because the decoder cross-checks them.
        """
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
            MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY,
        )

        batch.extra.pop(MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY, None)
        cls._sync_canonical_task(batch, "t2va")
        shape = dict(plan.shape)
        shape["frame_count"] = chunk_plan.published_frames
        shape["video_latent_t"] = int(batch.latents.shape[2])
        shape["audio_latent_t"] = int(batch.audio_latents.shape[-1])
        batch.extra[MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY] = msgspec.structs.replace(
            plan, task="t2va", shape=shape
        )
        batch.raw_latent_shape = tuple(batch.latents.shape)

    # ---- chunk state ----

    def _resolve_chunk_prompts(
        self, batch: Req, plan, chunk_count: int
    ) -> list[str]:
        chunk_prompts = getattr(batch.sampling_params, "chunk_prompts", None)
        if chunk_prompts is None:
            return [plan.prompt] * chunk_count
        if len(chunk_prompts) != chunk_count:
            raise ValueError(
                "chunk_prompts must carry one prompt per chunk: expected "
                f"{chunk_count}, got {len(chunk_prompts)}"
            )
        return list(chunk_prompts)

    def _precompute_text_embeddings(
        self,
        batch: Req,
        server_args: ServerArgs,
        plan,
        prompts: list[str],
    ) -> list[Any]:
        """Encode every chunk prompt up front and keep the payloads resident.

        Encoding once per distinct prompt avoids paging the text encoder back
        in mid-loop, and the payloads are a few MB each.
        """
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
            MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY,
        )

        cached: dict[str, Any] = {}
        # The registered TextEncodingStage already encoded plan.prompt.
        existing = batch.extra.get(MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY)
        if existing is not None:
            cached[plan.prompt] = existing
        for prompt in prompts:
            if prompt in cached:
                continue
            batch.extra.pop(MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY, None)
            batch.extra[MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY] = msgspec.structs.replace(
                plan, prompt=prompt
            )
            self._text_encoding.forward(batch, server_args)
            payload = batch.extra.get(MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY)
            if payload is None:
                raise RuntimeError(
                    "MiniMax H3 text encoding produced no payload for a chunk prompt"
                )
            cached[prompt] = payload
        batch.extra[MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY] = plan
        return [cached[prompt] for prompt in prompts]

    @staticmethod
    def _sync_canonical_task(batch: Req, task: str) -> None:
        """Keep the canonical request's task aligned with the chunk plan.

        The decoding stage cross-checks the two and refuses to run when they
        disagree.
        """
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY,
        )

        canonical = batch.extra.get(MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY)
        if isinstance(canonical, dict):
            canonical["task"] = task

    # ---- join helpers ----

    @staticmethod
    def _to_host(tensor: torch.Tensor) -> torch.Tensor:
        """Copy a decode output to a plain CPU tensor.

        Decode results descend from tensors allocated inside the denoise
        stage's InferenceMode; copying into a tensor allocated here keeps the
        joined result usable outside that mode and bounds device memory to one
        chunk.
        """
        host = torch.empty(
            tuple(tensor.shape), dtype=torch.float32, device="cpu"
        )
        host.copy_(tensor)
        return host

    @staticmethod
    def _fit_audio(
        audio: torch.Tensor,
        *,
        published_frames: int,
        fps: int,
        sample_rate: int,
    ) -> torch.Tensor:
        """Trim or pad the joined waveform to the published video duration.

        Per-chunk rounding leaves the concatenated audio a few samples off the
        video timeline; the delivery contract checks A/V drift, so settle it
        against the frame count that is actually published.
        """
        target = int(round(published_frames / fps * sample_rate))
        length = int(audio.shape[2])
        if length == target:
            return audio
        if length > target:
            return audio[:, :, :target].contiguous()
        pad = torch.zeros(
            audio.shape[0],
            audio.shape[1],
            target - length,
            dtype=audio.dtype,
            device=audio.device,
        )
        return torch.cat((audio, pad), dim=2).contiguous()

    @staticmethod
    def _publish_resolved_length(batch: Req, *, published_frames: int) -> None:
        """Record the joined length on the worker-side request.

        Delivery validation reads its own copy in the API process (see
        ``MiniMaxH3SamplingParams.streaming_published_frames``); this keeps the
        worker's transport metadata honest for anything that saves from here.
        """
        batch.num_frames = published_frames
        if batch.sampling_params is not None:
            batch.sampling_params.streaming_published_frames = published_frames

    # ---- verification ----

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        return self._latent_preparation.verify_input(batch, server_args)

    def verify_output(
        self, batch: OutputBatch, server_args: ServerArgs
    ) -> VerificationResult:
        result = VerificationResult()
        result.add_check("output", batch.output, [V.is_tensor, V.with_dims(5)])
        result.add_check("audio", batch.audio, [V.is_tensor, V.with_dims(3)])
        result.add_check("audio_sample_rate", batch.audio_sample_rate, V.positive_int)
        return result


__all__ = ["MiniMaxH3StreamingChunkStage"]
