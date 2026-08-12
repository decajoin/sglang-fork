# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 short-edge tier contracts (360p - 1080p).

The 768 tier is the pre-existing public geometry and is pinned exactly; the
other tiers must apply the identical adapt_shape_v1 policy at their own scale.
"""

from __future__ import annotations

import pytest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    minimax_h3_validate_canonical_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_SUPPORTED_SHORT_EDGES,
    minimax_h3_max_pixels,
    minimax_h3_resolve_plan,
    minimax_h3_resolve_spatial_shape,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    MINIMAX_H3_FINITE_ASPECT_RATIOS,
)

# The shipped 768 canvases. These are the public contract and must not move.
_TIER_768_CANVASES = {
    "21:9": (1536, 672),
    "16:9": (1344, 768),
    "4:3": (1024, 768),
    "1:1": (768, 768),
    "3:4": (768, 1024),
    "9:16": (768, 1344),
}


def _ratio_pair(aspect_ratio: str) -> tuple[int, int]:
    width, height = aspect_ratio.split(":")
    return int(width), int(height)


@pytest.mark.parametrize(
    ("aspect_ratio", "expected"), sorted(_TIER_768_CANVASES.items())
)
def test_768_tier_geometry_is_unchanged(aspect_ratio, expected):
    ar_w, ar_h = _ratio_pair(aspect_ratio)
    shape = minimax_h3_resolve_spatial_shape(
        width=ar_w, height=ar_h, base_short_edge=768
    )
    assert (shape["width"], shape["height"]) == expected


def test_768_remains_the_default_short_edge():
    """Omitting base_short_edge must keep resolving the shipped 768 canvases."""

    for aspect_ratio, expected in _TIER_768_CANVASES.items():
        ar_w, ar_h = _ratio_pair(aspect_ratio)
        shape = minimax_h3_resolve_spatial_shape(width=ar_w, height=ar_h)
        assert (shape["width"], shape["height"]) == expected


@pytest.mark.parametrize("aspect_ratio", MINIMAX_H3_FINITE_ASPECT_RATIOS)
@pytest.mark.parametrize("omitted", [True, False])
def test_absent_short_edge_defaults_to_the_768_tier(aspect_ratio, omitted):
    """A target without short_edge must behave exactly like an explicit 768."""

    target = {"aspect_ratio": aspect_ratio, "duration_seconds": 5.0}
    if not omitted:
        target["short_edge"] = None
    canonical = minimax_h3_validate_canonical_request(
        task="t2va", prompt="a raccoon", conditions=[], target=target
    )
    assert canonical["target"]["short_edge"] == 768

    explicit = minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="a raccoon",
        conditions=[],
        target={
            "short_edge": 768,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": 5.0,
        },
    )
    default_shape = minimax_h3_resolve_plan(canonical).shape
    explicit_shape = minimax_h3_resolve_plan(explicit).shape
    assert default_shape == explicit_shape
    assert (default_shape["width"], default_shape["height"]) == _TIER_768_CANVASES[
        aspect_ratio
    ]


@pytest.mark.parametrize("short_edge", MINIMAX_H3_SUPPORTED_SHORT_EDGES)
@pytest.mark.parametrize("aspect_ratio", MINIMAX_H3_FINITE_ASPECT_RATIOS)
def test_every_tier_is_grid_aligned(short_edge, aspect_ratio):
    ar_w, ar_h = _ratio_pair(aspect_ratio)
    shape = minimax_h3_resolve_spatial_shape(
        width=ar_w, height=ar_h, base_short_edge=short_edge
    )
    assert shape["width"] % MINIMAX_H3_CANVAS_MULTIPLE == 0
    assert shape["height"] % MINIMAX_H3_CANVAS_MULTIPLE == 0
    assert shape["base_short_edge"] == short_edge
    assert shape["max_pixels"] == minimax_h3_max_pixels(short_edge)


@pytest.mark.parametrize("short_edge", MINIMAX_H3_SUPPORTED_SHORT_EDGES)
@pytest.mark.parametrize("aspect_ratio", MINIMAX_H3_FINITE_ASPECT_RATIOS)
def test_tier_area_stays_within_one_grid_step_of_the_cap(short_edge, aspect_ratio):
    """Nearest-grid rounding may exceed the soft cap, but only marginally."""

    ar_w, ar_h = _ratio_pair(aspect_ratio)
    shape = minimax_h3_resolve_spatial_shape(
        width=ar_w, height=ar_h, base_short_edge=short_edge
    )
    cap = minimax_h3_max_pixels(short_edge)
    area = shape["width"] * shape["height"]
    if shape["size_mode"] == "short_edge":
        assert area <= cap
    else:
        # One 32px step on the longer axis bounds the post-rounding overshoot.
        slack = MINIMAX_H3_CANVAS_MULTIPLE * max(shape["width"], shape["height"])
        assert area <= cap + slack


@pytest.mark.parametrize("aspect_ratio", MINIMAX_H3_FINITE_ASPECT_RATIOS)
def test_canvas_area_is_monotonic_in_the_tier(aspect_ratio):
    ar_w, ar_h = _ratio_pair(aspect_ratio)
    areas = []
    for short_edge in sorted(MINIMAX_H3_SUPPORTED_SHORT_EDGES):
        shape = minimax_h3_resolve_spatial_shape(
            width=ar_w, height=ar_h, base_short_edge=short_edge
        )
        areas.append(shape["width"] * shape["height"])
    assert areas == sorted(areas)
    assert areas[0] < areas[-1]


@pytest.mark.parametrize("short_edge", MINIMAX_H3_SUPPORTED_SHORT_EDGES)
def test_tier_survives_validation_and_plan_resolution(short_edge):
    canonical = minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="a raccoon",
        conditions=[],
        target={
            "short_edge": short_edge,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
        },
    )
    assert canonical["target"]["short_edge"] == short_edge
    shape = minimax_h3_resolve_plan(canonical).shape
    assert shape["base_short_edge"] == short_edge
    assert shape["geometry"] == "resolved_v2"
    assert shape["width"] % MINIMAX_H3_CANVAS_MULTIPLE == 0
    assert shape["height"] % MINIMAX_H3_CANVAS_MULTIPLE == 0


@pytest.mark.parametrize("short_edge", [700, 1440, 0, -768, 767])
def test_unsupported_tiers_are_rejected(short_edge):
    with pytest.raises(ValueError, match="short_edge must be one of"):
        minimax_h3_validate_canonical_request(
            task="t2va",
            prompt="a raccoon",
            conditions=[],
            target={
                "short_edge": short_edge,
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
        )


@pytest.mark.parametrize("short_edge", [768.0, "768", True])
def test_non_integer_tiers_are_rejected(short_edge):
    with pytest.raises(ValueError, match="short_edge must be an integer"):
        minimax_h3_validate_canonical_request(
            task="t2va",
            prompt="a raccoon",
            conditions=[],
            target={
                "short_edge": short_edge,
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
        )


@pytest.mark.parametrize("short_edge", MINIMAX_H3_SUPPORTED_SHORT_EDGES)
def test_probed_display_ratios_follow_the_requested_tier(short_edge):
    """fl2va/ref2va derive geometry from probed material at the same tier."""

    shape = minimax_h3_resolve_spatial_shape(
        width=1920.0, height=1080.0, base_short_edge=short_edge
    )
    from_ratio = minimax_h3_resolve_spatial_shape(
        width=16, height=9, base_short_edge=short_edge
    )
    assert (shape["width"], shape["height"]) == (
        from_ratio["width"],
        from_ratio["height"],
    )
