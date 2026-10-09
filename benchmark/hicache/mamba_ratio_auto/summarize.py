"""Summarize raw A/B request records, startup memory, and fault injection logs."""

import argparse
import json
import re
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--normal", type=Path, required=True)
p.add_argument("--slow", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()


def stats(rows):
    times = sorted(x["ttft"] for x in rows)
    tokens = sum(x["input"] for x in rows)
    return dict(
        n=len(rows),
        mean=statistics.mean(times),
        p50=statistics.median(times),
        p95=times[max(0, int(len(times) * 0.95 + 0.999999) - 1)],
        maximum=max(times),
        tokens=tokens,
        **{
            k: sum(x[k] for x in rows)
            for k in ("device", "host", "storage", "recompute")
        },
    )


summary = {}
for label in ("base", "auto"):
    normal = [
        json.loads(x) for x in (a.normal / (label + ".jsonl")).read_text().splitlines()
    ]
    slow = [
        json.loads(x) for x in (a.slow / (label + ".jsonl")).read_text().splitlines()
    ]
    pick = lambda exp: [x for x in normal if x["experiment"] == exp]
    e1 = pick("E1_return")
    e2 = pick("E2_multiturn")
    e3 = [x for x in slow if x["experiment"] == "E3_return"]
    item = dict(
        E1=stats(e1),
        E1_rows=e1,
        cold=stats(pick("E1_cold") + pick("E1_filler")),
        E2=stats([x for x in e2 if x["turn"] > 0]),
        E2_turns={str(i): stats([x for x in e2 if x["turn"] == i]) for i in range(3)},
        E3=stats(e3),
        E3_rows=e3,
        E3_hit_ratio=sum(x["storage"] for x in e3) / (6 * 131072),
        drain=[x for x in slow if x["experiment"] == "E3_drain"],
        sanity=pick("sanity_cold") + pick("sanity_l1"),
    )
    logs = (a.normal / (label + "_server.log")).read_text()
    alloc = re.findall(
        r"Allocating (?:\w+ hierarchical KV host pool: \d+ tokens, |)([\d.]+) GB host memory",
        logs,
    )
    # Startup logs print each pool once per TP rank.
    item["host_alloc_log_GB_per_rank"] = dict(
        main_kv=float(alloc[0]), mamba=float(alloc[2]), draft_kv=float(alloc[4])
    )
    trace = (a.slow / (label + "_server.log")).read_text()
    counters = re.findall(
        r"TP(\d).*P0TRACE alloc_host mamba n=(\d+) fail=(\d+).*min_avail=(\d+)", trace
    )
    item["allocation_trace"] = {
        str(rank): dict(
            sampled_calls=max(int(x[1]) for x in counters if int(x[0]) == rank),
            failures=max(int(x[2]) for x in counters if int(x[0]) == rank),
            min_available_observed=min(
                int(x[3]) for x in counters if int(x[0]) == rank
            ),
        )
        for rank in (0, 1)
        if any(int(x[0]) == rank for x in counters)
    }
    memory = json.loads((a.normal / "memory.json").read_text())[label]
    item["process_PSS_GB"] = sum(x["Pss_kB"] for x in memory.values()) * 1024 / 1e9
    item["scheduler_RSS_GB"] = sorted(
        (x["VmRSS_kB"] * 1024 / 1e9 for x in memory.values()), reverse=True
    )[:2]
    summary[label] = item
summary["validation"] = dict(
    auto_E1_full_L2=all(
        x["device"] == 0 and x["host"] == 131072 and x["storage"] == 0
        for x in summary["auto"]["E1_rows"]
    ),
    auto_E3_full_L3=summary["auto"]["E3_hit_ratio"] == 1,
    auto_E2_lower_mean=summary["auto"]["E2"]["mean"] < summary["base"]["E2"]["mean"],
    all_requests_completed_4_tokens=all(
        x.get("completion_tokens") == 4
        for phase in (a.normal, a.slow)
        for label in ("base", "auto")
        for x in [
            json.loads(line)
            for line in (phase / (label + ".jsonl")).read_text().splitlines()
        ]
        if "ttft" in x
    ),
)
a.out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
