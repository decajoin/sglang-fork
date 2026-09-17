# SPDX-License-Identifier: Apache-2.0
"""Persistent clean media K/V shared across MiniMax H3 streaming chunks.

A streaming continuation inherits its past as attention K/V instead of as a
re-encoded pixel frame. The K/V staged here are the ones the DiT produced on a
dedicated sigma=0 forward over the finished chunk, so they describe clean
media rather than any intermediate denoising state.

Rows are captured after the Ulysses all-to-all, where each rank holds the full
packed sequence and its own head shard, and after RoPE, which is applied to K
before that exchange. Cached keys therefore already carry their absolute clip
positions and need no rotation when a later chunk attends to them.

Only target media rows are kept: text, padding and condition rows belong to the
chunk that produced them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

MINIMAX_H3_VIDEO_TOKEN_TAG = 0
MINIMAX_H3_AUDIO_TOKEN_TAG = 2


@dataclass(frozen=True)
class MiniMaxH3KVContract:
    """Shape every cached layer must satisfy, learned from the first stage."""

    local_heads: int
    head_dim: int
    dtype: torch.dtype
    device_type: str


class MiniMaxH3StreamingKVCache:
    """Transactional per-layer store of clean media K/V.

    A commit is all-or-nothing: a chunk that fails part-way through the block
    stack leaves the history exactly as the previous chunk left it, because a
    half-written history would silently corrupt every chunk after it.
    """

    # Main DiT stack only. The two token-refiner blocks share the attention
    # class but run the text sequence with no RoPE, so their keys carry no
    # clip position and are not history.
    _MAIN_LAYER_PATTERN = re.compile(r"^blocks\.\d+\.attn$")

    def __init__(self) -> None:
        self._layer_names: tuple[str, ...] = ()
        self._history: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._staged: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._staged_tags: torch.Tensor | None = None
        self._commit_tags: list[torch.Tensor] = []
        self._contract: MiniMaxH3KVContract | None = None
        self._commit_index = 0

    # ---- introspection ----

    @property
    def layer_names(self) -> tuple[str, ...]:
        """Layers recorded by the first commit; empty until one happens."""
        return self._layer_names

    @classmethod
    def accepts(cls, layer_name: str) -> bool:
        """Whether this layer's K/V belong in the clip history."""
        return bool(cls._MAIN_LAYER_PATTERN.match(layer_name))

    @property
    def contract(self) -> MiniMaxH3KVContract | None:
        return self._contract

    @property
    def committed_chunks(self) -> int:
        return self._commit_index

    @property
    def commit_active(self) -> bool:
        return self._staged_tags is not None

    @property
    def history_tokens(self) -> int:
        if not self._history:
            return 0
        return int(next(iter(self._history.values()))[0].shape[0])

    @property
    def history_video_tokens(self) -> int:
        return self._tag_total(MINIMAX_H3_VIDEO_TOKEN_TAG)

    @property
    def history_audio_tokens(self) -> int:
        return self._tag_total(MINIMAX_H3_AUDIO_TOKEN_TAG)

    def _tag_total(self, tag: int) -> int:
        return sum(int((tags == tag).sum()) for tags in self._commit_tags)

    def resident_bytes(self) -> int:
        return sum(
            int(k.numel() * k.element_size() + v.numel() * v.element_size())
            for k, v in self._history.values()
        )

    def history(self, layer_name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Clean K/V from every retained earlier chunk, or None when empty."""
        return self._history.get(layer_name)

    # ---- commit transaction ----

    def begin_commit(self, token_tags: torch.Tensor) -> None:
        """Open a commit for rows tagged by ``token_tags``.

        ``token_tags`` describes the rows every layer is about to stage, in the
        order they will be staged.
        """
        if self.commit_active:
            raise RuntimeError("a MiniMax H3 streaming KV commit is already active")
        if token_tags.ndim != 1 or token_tags.numel() == 0:
            raise ValueError("commit token tags must be a non-empty 1-D tensor")
        tags = token_tags.detach().to(device="cpu", dtype=torch.long, copy=True)
        unknown = tags[
            (tags != MINIMAX_H3_VIDEO_TOKEN_TAG) & (tags != MINIMAX_H3_AUDIO_TOKEN_TAG)
        ]
        if unknown.numel():
            raise ValueError(
                "a streaming KV commit takes only video and audio rows, got tag "
                f"{int(unknown[0])}"
            )
        self._staged_tags = tags
        self._staged.clear()

    def stage(self, layer_name: str, key: torch.Tensor, value: torch.Tensor) -> None:
        """Record one layer's clean K/V for the open commit."""
        if not self.commit_active:
            raise RuntimeError("begin_commit() must precede stage()")
        if not self.accepts(layer_name):
            raise KeyError(f"unexpected MiniMax H3 attention layer: {layer_name}")
        if layer_name in self._staged:
            raise RuntimeError(f"{layer_name}: K/V staged more than once")
        if key.ndim != 3 or key.shape != value.shape:
            raise ValueError(
                f"{layer_name}: K/V must be matching [tokens, heads, head_dim], got "
                f"{tuple(key.shape)} and {tuple(value.shape)}"
            )
        assert self._staged_tags is not None
        if int(key.shape[0]) != int(self._staged_tags.numel()):
            raise ValueError(
                f"{layer_name}: staged {int(key.shape[0])} rows but the commit "
                f"describes {int(self._staged_tags.numel())}"
            )
        contract = MiniMaxH3KVContract(
            local_heads=int(key.shape[1]),
            head_dim=int(key.shape[2]),
            dtype=key.dtype,
            device_type=key.device.type,
        )
        if self._contract is None:
            self._contract = contract
        elif contract != self._contract:
            raise ValueError(
                f"{layer_name}: K/V shape {contract} does not match the cache "
                f"contract {self._contract}"
            )
        self._staged[layer_name] = (key.detach(), value.detach())

    def commit(self) -> None:
        """Append every staged layer to the history, atomically."""
        if not self.commit_active:
            raise RuntimeError("no MiniMax H3 streaming KV commit is active")
        staged = tuple(sorted(self._staged, key=_layer_sort_key))
        if not staged:
            raise RuntimeError("a streaming KV commit staged no attention layers")
        if not self._layer_names:
            # The first commit defines the stack; every later one must match it
            # exactly, or the history would describe different layers at
            # different depths.
            self._layer_names = staged
        elif staged != self._layer_names:
            missing = [name for name in self._layer_names if name not in self._staged]
            raise RuntimeError(
                f"a streaming KV commit staged {len(staged)} attention layers but "
                f"the history has {len(self._layer_names)}"
                + (f", first missing {missing[0]}" if missing else "")
            )
        # One layer at a time: appending all fifty at once would hold two whole
        # histories on the device at the peak.
        for layer_name in self._layer_names:
            key, value = self._staged.pop(layer_name)
            previous = self._history.pop(layer_name, None)
            if previous is None:
                self._history[layer_name] = (key, value)
            else:
                self._history[layer_name] = (
                    torch.cat((previous[0], key), dim=0),
                    torch.cat((previous[1], value), dim=0),
                )
        assert self._staged_tags is not None
        self._commit_tags.append(self._staged_tags)
        self._commit_index += 1
        self._clear_staging()

    def rollback(self) -> None:
        self._clear_staging()

    def clear(self) -> None:
        self._history.clear()
        self._commit_tags.clear()
        self._commit_index = 0
        self._clear_staging()

    def _clear_staging(self) -> None:
        self._staged.clear()
        self._staged_tags = None

    # ---- retention ----

    def retain_sink_and_recent(
        self, recent_chunks: int = 1, *, video_only_sink: bool = True
    ) -> None:
        """Keep the first chunk as a sink and the last ``recent_chunks`` whole.

        The first chunk is what the clip looked like before any drift, so it
        stays as a long-term appearance sink. Chunks in between are dropped,
        which is what bounds the history.

        ``video_only_sink`` drops the sink's audio. That halves the sink but
        leaves the retained history holding video rows whose audio partners are
        gone, which an audio-video packed sequence has never seen in training.
        """
        if self.commit_active:
            raise RuntimeError("cannot trim streaming KV during an active commit")
        if recent_chunks < 0:
            raise ValueError("recent_chunks must not be negative")
        chunk_count = len(self._commit_tags)
        if chunk_count <= recent_chunks:
            return
        recent_start = max(1, chunk_count - recent_chunks)
        selection: list[tuple[int, bool]] = [(0, video_only_sink)]
        selection.extend((index, False) for index in range(recent_start, chunk_count))
        self._retain(selection)

    def drop_audio_history(self) -> int:
        """Remove every audio row, keeping all retained video rows."""
        if self.commit_active:
            raise RuntimeError("cannot trim streaming KV during an active commit")
        removed = self.history_audio_tokens
        if not self._commit_tags:
            return 0
        self._retain([(index, True) for index in range(len(self._commit_tags))])
        return removed

    def _retain(self, selection: list[tuple[int, bool]]) -> None:
        offsets = [0]
        for tags in self._commit_tags:
            offsets.append(offsets[-1] + int(tags.numel()))

        rows: list[int] = []
        kept_tags: list[torch.Tensor] = []
        for chunk_index, video_only in selection:
            tags = self._commit_tags[chunk_index]
            start = offsets[chunk_index]
            local = (
                torch.nonzero(tags == MINIMAX_H3_VIDEO_TOKEN_TAG, as_tuple=False).view(-1)
                if video_only
                else torch.arange(tags.numel())
            )
            if not local.numel():
                continue
            rows.extend((start + local).tolist())
            kept_tags.append(tags.index_select(0, local))

        if not rows:
            self._history.clear()
            self._commit_tags.clear()
            return
        # One layer at a time, and the old tensor has to lose its last reference
        # before the next one is allocated. Holding ``self._history.items()`` in
        # a list instead pins all 50 layers of the old history for the whole
        # loop, which makes trimming cost twice the cache -- the peak that puts
        # 768p over an H200.
        index: torch.Tensor | None = None
        for layer_name in list(self._history):
            key, value = self._history.pop(layer_name)
            if index is None:
                index = torch.tensor(rows, dtype=torch.long, device=key.device)
            trimmed_key = key.index_select(0, index)
            del key
            trimmed_value = value.index_select(0, index)
            del value
            self._history[layer_name] = (trimmed_key, trimmed_value)
        self._commit_tags = kept_tags


@dataclass
class MiniMaxH3StreamingChunkContext:
    """Where one chunk sits on the clip, and the history it reads and writes.

    ``media_time_origin`` and ``renorm_anchor`` are filled in by the first
    chunk and read by every later one, so this object is mutated as the chain
    advances rather than rebuilt.
    """

    cache: MiniMaxH3StreamingKVCache
    chunk_index: int
    video_latent_index_origin: int
    audio_latent_index_origin: int
    recent_chunks: int = 1
    video_only_sink: bool = True
    # Frozen at the first chunk's text length; every chunk's media coordinates
    # are measured from it so they never restart.
    media_time_origin: float | None = None
    # Per-column mean/std of the first chunk's clean video rows. Later chunks
    # are mapped onto it, which stops a slow statistical drift from riding the
    # chain the way the pixel-anchor handoff let it.
    renorm_anchor: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def is_first_chunk(self) -> bool:
        return self.chunk_index == 0


def minimax_h3_renormalize_video_rows(
    rows: torch.Tensor,
    context: MiniMaxH3StreamingChunkContext,
) -> torch.Tensor:
    """Map a chunk's clean video rows onto the first chunk's statistics.

    The first chunk defines the anchor and passes through untouched.
    """
    current = rows.detach().to(torch.float32)
    mean = current.mean(dim=0, keepdim=True)
    std = current.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    if context.renorm_anchor is None:
        context.renorm_anchor = (mean, std)
        return rows
    anchor_mean, anchor_std = context.renorm_anchor
    normalized = (current - mean).div(std).mul(anchor_std.to(current.device)).add(
        anchor_mean.to(current.device)
    )
    return normalized.to(dtype=rows.dtype)


def minimax_h3_streaming_anchor_rows(
    latents: torch.Tensor,
    *,
    latent_index_origin: int,
) -> torch.Tensor:
    """This chunk's last single-frame latent, as fl2va condition rows.

    Under the (1, 4, 4, 4, 4) weighting a latent whose *global* index is a
    multiple of five spans exactly one frame; the others span four and are not
    a still. Taking the single-frame one keeps the anchor the quantity a
    first-frame condition is defined to be.

    The rows come straight from the latent the DiT produced, in the same packed
    row space the condition rows live in, so the anchor never leaves that space
    -- no decode, no re-encode, and none of the per-hop photometric gain that
    round trip applies.
    """
    if latents.ndim != 5 or int(latents.shape[0]) != 1:
        raise ValueError(
            f"streaming anchor needs [1, C, T, H, W] latents, got {tuple(latents.shape)}"
        )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
        minimax_h3_patchify_video_latent,
    )

    count = int(latents.shape[2])
    offsets = [
        index
        for index in range(count)
        if (int(latent_index_origin) + index) % 5 == 0
    ]
    if not offsets:
        raise ValueError(
            "a streaming chunk must contain at least one single-frame latent"
        )
    index = offsets[-1]
    return minimax_h3_patchify_video_latent(
        latents[:, :, index : index + 1], patch_size=[1, 2, 2]
    )


def _layer_sort_key(layer_name: str) -> int:
    return int(layer_name.split(".")[1])


__all__ = [
    "MINIMAX_H3_AUDIO_TOKEN_TAG",
    "MiniMaxH3StreamingChunkContext",
    "minimax_h3_renormalize_video_rows",
    "minimax_h3_streaming_anchor_rows",
    "MINIMAX_H3_VIDEO_TOKEN_TAG",
    "MiniMaxH3KVContract",
    "MiniMaxH3StreamingKVCache",
]
