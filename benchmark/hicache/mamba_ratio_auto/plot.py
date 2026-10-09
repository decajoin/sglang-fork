"""Plot the measured cache benefit and the Host memory cost."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--summary", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
s = json.loads(a.summary.read_text())
labels = ["Default", "Auto"]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), layout="constrained")
values = [
    [s[k]["E1"]["mean"] for k in ("base", "auto")],
    [s[k]["E2"]["mean"] for k in ("base", "auto")],
    [100 * s[k]["E3_hit_ratio"] for k in ("base", "auto")],
    [(988928 * 16384 + 989184 * 5120 + n * 39223296) / 1e9 for n in (81, 251)],
]
for ax, v, title, unit in zip(
    axes,
    values,
    ["E1: L2 return", "E2: 24 returns", "E3: delayed L3 ack", "Host cache / rank"],
    ["Mean TTFT (s)", "Mean TTFT (s)", "L3 prefix recovered (%)", "Cache tensors (GB)"],
):
    bars = ax.bar(labels, v, color=["#8c9ba8", "#217a74"], width=0.6)
    ax.bar_label(bars, labels=[f"{x:.2f}" for x in v], padding=4)
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.set_ylim(0, max(v) * 1.25)
    if unit == "L3 prefix recovered (%)":
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
fig.suptitle("Agnes TP=2: default Mamba Host sizing vs auto (2026-10-09)", fontsize=12)
fig.savefig(a.out)
fig.savefig(a.out.with_suffix(".png"), dpi=160)
plt.close(fig)
