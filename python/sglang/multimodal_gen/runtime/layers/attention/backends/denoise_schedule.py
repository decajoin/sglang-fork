# SPDX-License-Identifier: Apache-2.0
"""The running denoise schedule's length, published for sparse backends.

A sparse backend that keeps the *last* few denoise steps dense cannot identify
the last step on its own: ``current_timestep`` is a zero-based counter with no
upper bound attached to it. The stage that owns the sigma schedule is the only
place the count is authoritative, so it publishes it here and every backend
that needs it reads it from here.

``sampling_params.num_inference_steps`` is a request-level *hint* and only a
fallback: for MiniMax-H3 it may be a ``(video, audio)`` pair rather than an
int, and the loop actually runs ``len(sigmas_video) - 1``. A backend that read
the hint instead would leave its tail cutoff silently inactive on exactly the
model these backends target -- an experiment that never ran looks like an
experiment that came back negative.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_denoise_total_steps: ContextVar[int | None] = ContextVar(
    "denoise_total_steps", default=None
)


@contextmanager
def denoise_total_steps(total: int | None) -> Iterator[None]:
    """Publish how many denoise steps the loop about to run will take.

    A no-op for every backend that does not read it.
    """
    token = _denoise_total_steps.set(total)
    try:
        yield
    finally:
        _denoise_total_steps.reset(token)


def get_denoise_total_steps(context: Any = None) -> int | None:
    """Length of the running denoise schedule, or None if nothing said.

    The published value wins. ``context`` is an optional forward context whose
    ``forward_batch.sampling_params.num_inference_steps`` is consulted only
    when it really is a positive int.
    """
    published = _denoise_total_steps.get()
    if isinstance(published, int) and published > 0:
        return published
    batch = getattr(context, "forward_batch", None)
    hint = getattr(getattr(batch, "sampling_params", None), "num_inference_steps", None)
    if isinstance(hint, int) and not isinstance(hint, bool) and hint > 0:
        return hint
    return None


__all__ = ["denoise_total_steps", "get_denoise_total_steps"]
