"""Local A/B experiment. Run with the repo venv; requires four idle GPUs.

The slow-L3 phase requires fault_injection.patch applied to the controller.
Logs and JSON are saved even when a request fails; all child groups are stopped.
"""

import argparse
import concurrent.futures
import json
import os
import random
import re
import signal
import subprocess
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PYTHON = str(REPO / ".venv/bin/python")
p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--phase", choices=["normal", "slow"], default="normal")
a = p.parse_args()
a.out = a.out.resolve()
a.out.mkdir(parents=True, exist_ok=True)
if any(a.out.iterdir()):
    raise ValueError("Use a new, empty output directory.")
if a.phase == "slow":
    source = (
        REPO / "python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py"
    )
    if "P0_L3_TOKENS_PER_S" not in source.read_text():
        raise RuntimeError("Apply fault_injection.patch before the slow phase.")
children = []
rows = []


def spawn(cmd, log, env=None):
    f = (a.out / log).open("w")
    proc = subprocess.Popen(
        cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
        cwd=REPO,
    )
    f.close()
    children.append(proc)
    return proc


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if proc.poll() is None:
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)


def wait_ready(url, procs, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(x.poll() is not None for x in procs):
            raise RuntimeError(f"Child exited before {url} became ready; see logs")
        try:
            if requests.get(url, timeout=3).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(url)


def gen(url, ids):
    start = time.perf_counter()
    ttft = None
    meta = None
    with requests.post(
        url + "/generate",
        json={
            "input_ids": ids,
            "stream": True,
            "sampling_params": {
                "max_new_tokens": 4,
                "ignore_eos": True,
                "temperature": 0,
            },
        },
        stream=True,
        timeout=1200,
    ) as response:
        response.raise_for_status()
        # requests' default 512-byte buffer can bias TTFT on tiny SSE responses.
        for line in response.iter_lines(chunk_size=1):
            if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                payload = json.loads(line[5:])
                if "error" in payload:
                    raise RuntimeError(payload["error"])
                if ttft is None:
                    ttft = time.perf_counter() - start
                meta = payload["meta_info"]
    if ttft is None or meta is None:
        raise RuntimeError("No streamed generation received")
    det = meta.get("cached_tokens_details") or {}
    dev, host, storage = (det.get(k, 0) for k in ("device", "host", "storage"))
    return dict(
        input=len(ids),
        ttft=ttft,
        device=dev,
        host=host,
        storage=storage,
        recompute=len(ids) - dev - host - storage,
        output_ids=meta.get("output_token_ids"),
        output_text=payload.get("text"),
        completion_tokens=meta.get("completion_tokens"),
    )


def flush(url):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if requests.post(url + "/flush_cache", timeout=30).status_code == 200:
            time.sleep(2)
            return
        time.sleep(2)
    raise TimeoutError("flush_cache")


def rand(rng, n):
    return [rng.randint(1000, 150000) for _ in range(n)]


def record(label, experiment, data, **extra):
    row = dict(config=label, experiment=experiment, **extra, **data)
    rows.append(row)
    # One append per finished request retains partial results on interruption.
    with (a.out / (label + ".jsonl")).open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "output_ids"}), flush=True)
    return row


def workload(label, url, metrics):
    if a.phase == "normal":
        for seed, fillers, quiet in [(1101, 3, 0), (1102, 3, 20), (1103, 5, 20)]:
            flush(url)
            rng = random.Random(seed)
            x = rand(rng, 131072)
            record(label, "E1_cold", gen(url, x), seed=seed)
            for i in range(fillers):
                record(
                    label, "E1_filler", gen(url, rand(rng, 131072)), seed=seed, filler=i
                )
            time.sleep(quiet)
            row = record(
                label,
                "E1_return",
                gen(url, x + rand(rng, 256)),
                seed=seed,
                fillers=fillers,
                quiet=quiet,
            )
            if row["device"] != 0:
                raise AssertionError("E1 did not evict X from L1")
        flush(url)
        rng = random.Random(1201)
        histories = [rand(rng, 65536) for _ in range(12)]
        for turn in range(3):
            for session, history in enumerate(histories):
                if turn:
                    history.extend(rand(rng, 2048))
                record(
                    label, "E2_multiturn", gen(url, history), turn=turn, session=session
                )
        # Unpressured short prefix: L1 behavior and output length remain usable.
        flush(url)
        rng = random.Random(1301)
        short = rand(rng, 4096)
        record(label, "sanity_cold", gen(url, short))
        record(label, "sanity_l1", gen(url, short + rand(rng, 256)))
    else:
        flush(url)
        check_rng = random.Random(1399)
        short = rand(check_rng, 4096)
        cold, warm = gen(url, short), gen(url, short)
        if cold["output_text"] is None or cold["output_text"] != warm["output_text"]:
            raise AssertionError("Repeated short-prefix output differs")
        record(label, "sanity_same_input_cold", cold)
        record(label, "sanity_same_input_repeat", warm)
        flush(url)
        rng = random.Random(1401)
        prompts = [rand(rng, 131072) for _ in range(6)]
        start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(gen, url, x) for x in prompts]
            for i, f in enumerate(futures):
                record(label, "E3_cold", f.result(), prompt=i)
        last, stable = None, 0
        deadline = time.monotonic() + 900
        while stable < 4:
            if time.monotonic() > deadline:
                raise TimeoutError("L3 drain")
            text = requests.get(metrics, timeout=5).text
            match = re.search(r"^master_key_count\S*\s+([0-9.eE+]+)", text, re.M)
            if not match:
                raise RuntimeError("No master_key_count metric")
            keys = int(float(match.group(1)))
            stable = stable + 1 if keys == last else 0
            last = keys
            time.sleep(10)
        record(label, "E3_drain", dict(seconds=time.monotonic() - start, keys=last))
        flush(url)
        for i, x in enumerate(prompts):
            record(label, "E3_return", gen(url, x + rand(rng, 256)), prompt=i)


def memory_snapshot():
    data = {}
    for label, _, _, _, _, proc in servers:
        # Include parent and workers by following the process group.
        ps = subprocess.check_output(["ps", "-eo", "pid,pgid"], text=True)
        pids = [
            int(line.split()[0])
            for line in ps.splitlines()[1:]
            if int(line.split()[1]) == proc.pid
        ]
        data[label] = {}
        for pid in pids:
            try:
                status = Path(f"/proc/{pid}/status").read_text()
                smaps = Path(f"/proc/{pid}/smaps_rollup").read_text()
            except FileNotFoundError:
                continue
            fields = {}
            for text in (status, smaps):
                for key in ("VmRSS", "VmLck", "VmPin", "Rss", "Pss", "Locked"):
                    m = re.search(r"^" + key + r":\s+(\d+) kB", text, re.M)
                    if m:
                        fields[key + "_kB"] = int(m.group(1))
            data[label][str(pid)] = fields
    (a.out / "memory.json").write_text(json.dumps(data, indent=2))
    (a.out / "gpu.csv").write_text(
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv",
            ],
            text=True,
        )
    )


try:
    servers = []
    for label, gpus, port, master, metric in [
        ("base", "0,1", 30080, 50099, 9099),
        ("auto", "2,3", 30081, 50100, 9100),
    ]:
        mp = spawn(
            [
                str(REPO / ".venv/bin/mooncake_master"),
                f"--rpc_port={master}",
                "--rpc_address=127.0.0.1",
                f"--metrics_port={metric}",
                "--metrics_host=127.0.0.1",
            ],
            label + "_master.log",
        )
        wait_ready(f"http://127.0.0.1:{metric}/metrics", [mp], 30)
        sp = spawn(
            [PYTHON, str(HERE / "store.py"), "--master", f"127.0.0.1:{master}"],
            label + "_store.log",
        )
        time.sleep(3)
        env = dict(
            os.environ,
            MC_MASTER=f"127.0.0.1:{master}",
            P0_L3_TOKENS_PER_S="2400" if a.phase == "slow" else "0",
        )
        cmd = ["bash", str(HERE / "launch.sh"), gpus, str(port)]
        if label == "auto":
            cmd += ["--hicache-mamba-ratio", "auto"]
        proc = spawn(cmd, label + "_server.log", env)
        servers.append((label, gpus, port, master, metric, proc))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs = [
            pool.submit(wait_ready, f"http://127.0.0.1:{port}/health", [proc])
            for label, _, port, _, _, proc in servers
        ]
        for f in fs:
            f.result()
    memory_snapshot()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs = [
            pool.submit(
                workload,
                label,
                f"http://127.0.0.1:{port}",
                f"http://127.0.0.1:{metric}/metrics",
            )
            for label, _, port, _, metric, _ in servers
        ]
        for f in fs:
            f.result()
    (a.out / "results.json").write_text(json.dumps(rows, indent=2))
    print("COMPLETE", flush=True)
finally:
    for proc in reversed(children):
        stop(proc)
