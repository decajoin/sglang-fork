#!/bin/bash
# Drop FlashInfer's SM120 Sage block-sparse kernel beside an installed
# flashinfer-python, so VSA-H3 can run its `flashinfer` path.
#
# Why a script instead of `pip install`: the kernel landed after 0.6.17, and
# installing the repo at that commit fails against nvidia-cutlass-dsl 4.6.0 --
# flashinfer's `__init__` reaches into `gdn_kernels`, which wants a newer
# CuTe-DSL than the sparse kernels themselves do. The sparse subpackage depends
# on nothing but `flashinfer.api_logging`, so it can sit next to the released
# one under a different name. Nothing in the released package imports either
# `sparse` or `sparse_sm120`, so the existing install is untouched.
#
# The copy is not recorded in any dist-info, which cuts both ways: `pip install
# -U flashinfer-python` leaves it behind stale. Re-run this script after any
# FlashInfer upgrade; it replaces the directory rather than merging into it.
#
# Idempotent. Override PYTHON=, FLASHINFER_COMMIT=, or SRC= as needed.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-6a84331e}"
FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/flashinfer-ai/flashinfer.git}"
# A checkout to copy from. Empty means clone a throwaway one.
SRC="${SRC:-}"

die() { echo "error: $*" >&2; exit 1; }

command -v git >/dev/null || die "git not found"
command -v "$PYTHON" >/dev/null || die "python not found: $PYTHON (set PYTHON=)"

# Where the released flashinfer lives. Also proves it is importable at all --
# the copy is useless without `flashinfer.api_logging` beside it.
PKG_DIR="$("$PYTHON" - <<'PY'
import os
import flashinfer
print(os.path.dirname(flashinfer.__file__))
PY
)" || die "cannot import flashinfer with ${PYTHON} (activate the venv, or set PYTHON=/path/to/venv/bin/python)"

DEST="${PKG_DIR}/cute_dsl/sparse_sm120"
echo "flashinfer package: ${PKG_DIR}"

# A system-wide install is usually root-owned, and a plain user would find that
# out from `cp` -- which runs after the replace below has already emptied the
# directory it was replacing, turning a failed upgrade into a broken install.
# Checked here rather than there so a clone is not paid for first.
if [[ ! -w "$(dirname "$DEST")" ]]; then
    hint="sudo PYTHON=\"${PYTHON}\""
    [[ -n "$SRC" ]] && hint="${hint} SRC=\"${SRC}\""
    die "$(dirname "$DEST") is not writable by $(id -un); re-run as its owner, or: ${hint} $0"
fi

# The kernel is SM120-only and asserts head_dim 128. Warn rather than refuse:
# the copy is harmless on other cards, and a build host may have no GPU at all.
"$PYTHON" - <<'PY' || true
import torch

if not torch.cuda.is_available():
    print("note: no GPU visible here; VSA-H3 gates on capability at runtime")
elif torch.cuda.get_device_capability() != (12, 0):
    cap = torch.cuda.get_device_capability()
    print(f"note: device is sm_{cap[0]}{cap[1]}, not sm_120 -- VSA-H3 will stay on Triton")
PY

CLEANUP=""
if [[ -z "$SRC" ]]; then
    SRC="$(mktemp -d)"
    CLEANUP="$SRC"
    trap '[[ -n "$CLEANUP" ]] && rm -rf "$CLEANUP"' EXIT
    echo "cloning ${FLASHINFER_REPO} @ ${FLASHINFER_COMMIT}"
    # Blobless rather than shallow: the commit is not a branch tip, so a
    # depth-1 clone cannot reach it without a fetch-by-sha the server may refuse.
    git clone --filter=blob:none --no-checkout "$FLASHINFER_REPO" "$SRC" >/dev/null 2>&1 \
        || die "clone failed"
    git -C "$SRC" checkout --quiet "$FLASHINFER_COMMIT" || die "checkout ${FLASHINFER_COMMIT} failed"
fi

SPARSE="${SRC}/flashinfer/cute_dsl/sparse"
[[ -f "${SPARSE}/bsa_attn_sm120.py" ]] \
    || die "${SPARSE}/bsa_attn_sm120.py missing -- wrong commit or wrong SRC"

# Replace, never merge: a leftover file from an older commit would shadow its
# replacement and the source fingerprint would happily cache against it.
rm -rf "$DEST"
cp -r "$SPARSE" "$DEST"
find "$DEST" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "installed: ${DEST}"

# Import the two modules VSA-H3 actually reaches for. This is the check that
# matters -- it is where a CuTe-DSL mismatch surfaces, not at copy time.
"$PYTHON" - <<'PY' || die "the copy does not import -- see the traceback above"
from flashinfer.cute_dsl.sparse_sm120.bsa_attn_sm120 import (
    bsa_attn_sm120_blk64_sage_fwd,  # noqa: F401
)
from flashinfer.cute_dsl.sparse_sm120.bsa_utils.sage_quant_sm120 import (  # noqa: F401
    quantize_sage_kv_sm120,
    quantize_sage_q_sm120,
)

print("ok: SM120 Sage kernel imports")
PY
