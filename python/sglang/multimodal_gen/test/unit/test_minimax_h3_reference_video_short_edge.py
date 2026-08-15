# SPDX-License-Identifier: Apache-2.0
"""Configurable ref2va reference-video short edge.

The released behaviour resizes every reference video to the request's own
target short-edge tier, so a 1080p job never conditions on a 768p re-encode.
The override exists only to trade conditioning fidelity for the cost of the
rows a reference video contributes -- it is by far the largest segment of the
packed sequence -- so the default path must stay exactly what it was.
"""

from __future__ import annotations

import pytest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.prequeue import (
    _configured_reference_video_short_edge,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
    minimax_h3_validate_reference_video_short_edge,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MINIMAX_H3_SUPPORTED_SHORT_EDGES,
    minimax_h3_resolve_spatial_shape,
)

# --------------------------------------------------------------------------
# the default must not move
# --------------------------------------------------------------------------


def test_unset_reader_returns_none_without_a_server():
    """No server (unit tests, offline resolver) means released behaviour."""
    assert _configured_reference_video_short_edge() is None


@pytest.mark.parametrize("target_tier", MINIMAX_H3_SUPPORTED_SHORT_EDGES)
def test_unset_follows_the_target_tier(target_tier):
    """Unset, a reference video resolves at the target's own tier."""
    shape = minimax_h3_resolve_spatial_shape(
        width=1920, height=1080, base_short_edge=target_tier
    )
    assert shape["base_short_edge"] == target_tier
    # The canvas snaps to the 32px grid, so 540 lands on 544; the tier is what
    # the policy records, the effective edge is that tier rounded.
    assert abs(min(shape["width"], shape["height"]) - target_tier) <= 32


# --------------------------------------------------------------------------
# the override
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tier", MINIMAX_H3_SUPPORTED_SHORT_EDGES)
def test_every_supported_tier_validates(tier):
    assert minimax_h3_validate_reference_video_short_edge(tier) == tier


@pytest.mark.parametrize(
    "value",
    (0, -540, 700, 1081, 32, "540", 540.5, None),
    ids=(
        "zero",
        "negative",
        "off-tier",
        "above-max",
        "multiple-of-32",
        "str",
        "float",
        "none",
    ),
)
def test_off_tier_values_are_rejected(value):
    """Restricted to the enumerable tier set, not an arbitrary grid."""
    with pytest.raises(ValueError):
        minimax_h3_validate_reference_video_short_edge(value)


def test_override_decouples_the_reference_from_the_target():
    """A 768 target with a 540 reference resolves each at its own tier."""
    target = minimax_h3_resolve_spatial_shape(
        width=1344, height=768, base_short_edge=768
    )
    reference = minimax_h3_resolve_spatial_shape(
        width=1920, height=1080, base_short_edge=540
    )
    assert target["base_short_edge"] == 768
    assert reference["base_short_edge"] == 540
    assert min(target["width"], target["height"]) == 768
    assert abs(min(reference["width"], reference["height"]) - 540) <= 32


def test_override_keeps_the_display_ratio():
    """Only the tier changes; the reference's own aspect must survive."""
    for tier in (1080, 768, 540, 480):
        shape = minimax_h3_resolve_spatial_shape(
            width=1920, height=1080, base_short_edge=tier
        )
        ratio = shape["width"] / shape["height"]
        assert abs(ratio - 16 / 9) < 0.06, (tier, shape["width"], shape["height"])


def test_lower_tier_really_cuts_the_row_count():
    """Rows scale with the area, which is what the flag is for.

    Latent rows are (h/8/2) * (w/8/2) per frame under the VAE's 8x spatial
    compression and 2x2 patchification, so they track pixel area exactly.
    """
    big = minimax_h3_resolve_spatial_shape(width=1920, height=1080, base_short_edge=768)
    small = minimax_h3_resolve_spatial_shape(
        width=1920, height=1080, base_short_edge=540
    )
    big_area = big["width"] * big["height"]
    small_area = small["width"] * small["height"]
    # (540/768)^2 = 0.494; allow for the 32px canvas rounding on both tiers.
    assert 0.45 < small_area / big_area < 0.55


def test_explicit_match_is_a_no_op():
    """Pinning the tier the target already uses must change nothing."""
    implicit = minimax_h3_resolve_spatial_shape(
        width=1920, height=1080, base_short_edge=768
    )
    explicit_same = minimax_h3_resolve_spatial_shape(
        width=1920,
        height=1080,
        base_short_edge=minimax_h3_validate_reference_video_short_edge(768),
    )
    assert implicit == explicit_same
