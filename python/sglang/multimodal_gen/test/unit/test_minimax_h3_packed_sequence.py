# SPDX-License-Identifier: Apache-2.0
"""Numerical contracts for MiniMax-H3 packed-sequence layouts."""

import unittest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)


class TestMiniMaxH3PackedSequence(unittest.TestCase):
    def test_t2va_structure(self):
        built = minimax_h3_packed_sequence(
            text_len=97,
            latent_t=62,
            latent_h=48,
            latent_w=76,
            audio_t=348,
            include_keyframe_cond=False,
        )
        self.assertTrue(built["update_mask"].all())
        self.assertEqual(int(built["img_pos"].shape[0]), 62 * 24 * 38)
        self.assertEqual(int(built["seq_len"]) % 64, 0)
        self.assertEqual(built["token_tags"][built["audio_pos"]].unique().tolist(), [2])

    def test_fl2va_first_last_cond_blocks_use_exact_rope_span(self):
        text_len = 11
        latent_t = 37
        built = minimax_h3_packed_sequence(
            text_len=text_len,
            latent_t=latent_t,
            latent_h=48,
            latent_w=76,
            audio_t=203,
            include_keyframe_cond=True,
            keyframe_frame_indices=[0, -1],
            frame_count=124,
        )
        frame_rows = 24 * 38
        cond_rows = 2 * frame_rows
        self.assertEqual(int((~built["update_mask"]).sum()), cond_rows)
        self.assertEqual(
            int(built["img_pos"].shape[0]),
            (2 + latent_t) * frame_rows,
        )
        cond_pos = built["img_pos"][:cond_rows].reshape(2, frame_rows)
        cond_t = [
            float(built["img_position_ids"][positions, 0].unique().item())
            for positions in cond_pos
        ]
        frame_rescale = 5.0 / 3.0
        temporal_span = sum(
            frame_rescale * (1, 4, 4, 4, 4)[index % 5] for index in range(latent_t)
        )
        self.assertEqual(cond_t[0], float(text_len))
        self.assertAlmostEqual(
            cond_t[1],
            float(text_len) + temporal_span - frame_rescale,
            places=12,
        )
        self.assertFalse(built["update_mask"][:cond_rows].any())
        self.assertTrue(built["update_mask"][cond_rows:].all())

    def test_i2va_and_l2va_single_cond_blocks_use_endpoint_rope(self):
        text_len = 11
        latent_t = 37
        frame_count = 124
        frame_rescale = 5.0 / 3.0
        temporal_span = sum(
            frame_rescale * (1, 4, 4, 4, 4)[index % 5] for index in range(latent_t)
        )
        for semantic_index, expected_t in (
            (0, float(text_len)),
            (-1, float(text_len) + temporal_span - frame_rescale),
        ):
            with self.subTest(semantic_index=semantic_index):
                built = minimax_h3_packed_sequence(
                    text_len=text_len,
                    latent_t=latent_t,
                    latent_h=48,
                    latent_w=76,
                    audio_t=203,
                    include_keyframe_cond=True,
                    keyframe_frame_indices=[semantic_index],
                    frame_count=frame_count,
                )
                frame_rows = 24 * 38
                self.assertEqual(int((~built["update_mask"]).sum()), frame_rows)
                positions = built["img_pos"][:frame_rows]
                cond_t = float(built["img_position_ids"][positions, 0].unique().item())
                self.assertAlmostEqual(cond_t, expected_t, places=12)

    def test_fl2va_keyframe_index_validation_is_defensive(self):
        common = dict(
            text_len=11,
            latent_t=37,
            latent_h=48,
            latent_w=76,
            audio_t=203,
            include_keyframe_cond=True,
            frame_count=124,
        )
        for frame_indices in (None, [1], [0, 52, -1]):
            with (
                self.subTest(frame_indices=frame_indices),
                self.assertRaises(ValueError),
            ):
                minimax_h3_packed_sequence(
                    **common,
                    keyframe_frame_indices=frame_indices,
                )

    def test_ref2va_structure(self):
        built = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=97,
            latent_t=112,
            latent_h=48,
            latent_w=84,
            audio_t=631,
            ref_blocks=[
                {"kind": "image", "latent_h": 64, "latent_w": 48},
                {"kind": "audio", "ref_audio_t": 582},
            ],
        )

        self.assertEqual(int(built["seq_len"]) % 64, 0)
        self.assertEqual(int((~built["update_mask"]).sum()), 32 * 24)
        self.assertEqual(int((~built["audio_update_mask"]).sum()), 582 * 2)
        self.assertEqual(built["token_tags"][built["audio_pos"]].unique().tolist(), [2])

    def test_ref2va_tags_every_picture_segment_with_its_grid(self):
        """Tile-based sparse attention needs each reference block's grid."""
        built = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=5,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=5,
            ref_blocks=[
                {"kind": "image", "latent_h": 8, "latent_w": 12},
                {
                    "kind": "video_audio",
                    "ref_audio_t": 3,
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 6,
                },
                {"kind": "audio", "ref_audio_t": 1},
            ],
        )
        segments = built["prefix_segments"]
        # text | still | ref audio | ref frames | ref audio | target audio
        self.assertEqual(segments, (5, 24, 6, 12, 2, 10))
        self.assertEqual(built["reference_visuals"], ((1, (1, 4, 6)), (3, (2, 2, 3))))
        self.assertEqual(built["video_grid"], (2, 2, 2))
        for index, grid in built["reference_visuals"]:
            self.assertEqual(grid[0] * grid[1] * grid[2], segments[index])

    def test_ref2va_empty_segments_do_not_shift_the_picture_tags(self):
        """A video reference with no audio drops a segment; tags must follow."""
        built = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=5,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=5,
            ref_blocks=[
                {
                    "kind": "video",
                    "ref_audio_t": 0,
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 6,
                },
            ],
        )
        segments = built["prefix_segments"]
        # text | ref frames | target audio -- the empty ref-audio segment is gone
        self.assertEqual(segments, (5, 12, 10))
        self.assertEqual(built["reference_visuals"], ((1, (2, 2, 3)),))
        for index, grid in built["reference_visuals"]:
            self.assertEqual(grid[0] * grid[1] * grid[2], segments[index])

    def test_ref2va_splits_the_text_block_around_its_pictures(self):
        """Qwen packs a vision block per reference image inside the prompt."""
        built = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=100,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=5,
            ref_blocks=[{"kind": "image", "latent_h": 8, "latent_w": 12}],
            text_visuals=[(10, 24, (1, 4, 6)), (40, 12, (1, 3, 4))],
        )
        segments = built["prefix_segments"]
        # The text block stays one segment: splitting it costs a partial tile
        # per boundary, and only a backend that sparsifies it should pay that.
        self.assertEqual(segments, (100, 24, 10))
        self.assertEqual(
            built["text_visuals"], ((10, 24, (1, 4, 6)), (40, 12, (1, 3, 4)))
        )
        self.assertEqual(built["reference_visuals"], ((1, (1, 4, 6)),))
        for start, rows, grid in built["text_visuals"]:
            self.assertEqual(grid[0] * grid[1] * grid[2], rows)
            self.assertLessEqual(start + rows, segments[0])

    def test_ref2va_text_pictures_stay_tagged_text_for_per_row_backends(self):
        """The split is structural; the per-row tags are the caller's to write."""
        built = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=100,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=5,
            ref_blocks=[{"kind": "image", "latent_h": 8, "latent_w": 12}],
            text_visuals=[(10, 24, (1, 4, 6))],
        )
        self.assertEqual(
            built["token_tags"][built["text_pos"]].unique().tolist(), [1]
        )

    def test_ref2va_text_pictures_may_touch_both_edges_of_the_block(self):
        """Spans that tile the whole block leave no protected gap at all."""
        built = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=36,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=5,
            ref_blocks=[],
            text_visuals=[(0, 12, (1, 3, 4)), (12, 24, (1, 4, 6))],
        )
        self.assertEqual(built["prefix_segments"], (36, 10))
        self.assertEqual(
            built["text_visuals"], ((0, 12, (1, 3, 4)), (12, 24, (1, 4, 6)))
        )

    def test_ref2va_publishes_every_temporal_block_of_a_video_reference(self):
        """Qwen packs one vision block per merged second, not one per video."""
        spans = [(4, 12, (1, 3, 4)), (20, 12, (1, 3, 4)), (36, 12, (1, 3, 4))]
        built = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=60,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=5,
            ref_blocks=[
                {
                    "kind": "video",
                    "ref_audio_t": 0,
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 6,
                }
            ],
            text_visuals=spans,
        )
        self.assertEqual(built["text_visuals"], tuple(spans))
        self.assertEqual(built["prefix_segments"], (60, 12, 10))
        self.assertEqual(built["reference_visuals"], ((1, (2, 2, 3)),))

    def test_ref2va_rejects_text_spans_that_do_not_describe_the_block(self):
        base = dict(
            text_len=36,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=5,
            ref_blocks=[],
        )
        with self.assertRaisesRegex(ValueError, "escapes"):
            minimax_h3_packed_sequence_ref2va_blocks(
                **base, text_visuals=[(30, 12, (1, 3, 4))]
            )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            minimax_h3_packed_sequence_ref2va_blocks(
                **base, text_visuals=[(0, 12, (1, 3, 4)), (6, 12, (1, 3, 4))]
            )
        with self.assertRaisesRegex(ValueError, "must cover rows"):
            minimax_h3_packed_sequence_ref2va_blocks(
                **base, text_visuals=[(0, 0, (1, 3, 4))]
            )
        with self.assertRaisesRegex(ValueError, "covers"):
            minimax_h3_packed_sequence_ref2va_blocks(
                **base, text_visuals=[(0, 12, (1, 3, 5))]
            )

    def test_ref2va_mixed_media_preserves_temporal_origin(self):
        built = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=5,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=5,
            ref_blocks=[
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {
                    "kind": "video_audio",
                    "ref_audio_t": 3,
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 4,
                },
                {"kind": "audio", "ref_audio_t": 1},
            ],
        )

        self.assertEqual(int((~built["update_mask"]).sum()), 12)
        self.assertEqual(int((~built["audio_update_mask"]).sum()), 8)
        target_video_t0 = built["img_position_ids"][built["img_pos"][12], 0]
        target_audio_t0 = built["img_position_ids"][built["audio_pos"][8], 0]
        self.assertEqual(float(target_audio_t0), float(target_video_t0))
