#!/usr/bin/env python3
"""Mine CI logs and rewrite MEMORY_CAPACITY_FLOORS in each e2e test file.

Parses engine allocation lines from GitHub Actions job logs (KV / SWA / Mamba /
DSV4), averages capacity metrics across recent runs, and writes floors at
``mean * 0.99`` as a module-level list in each test file::

    MEMORY_CAPACITY_FLOORS = [
        {"token_capacity": 52358, "kv_cache_gb": 6.39},
    ]

Runtime: ``popen_launch_server`` / PD health calls ``GET /server_info`` and
compares against the next unused floor (see ``sglang.test.memory_threshold``).

Usage:
    python3 scripts/ci/utils/update_memory_thresholds.py
    python3 scripts/ci/utils/update_memory_thresholds.py --dry-run
    python3 scripts/ci/utils/update_memory_thresholds.py --log-dir /tmp/ci_logs
    python3 scripts/ci/utils/update_memory_thresholds.py --run-id 29458283004

Requires ``gh`` authenticated against sgl-project/sglang for remote fetch.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.test.memory_threshold import (  # noqa: E402
    CAPACITY_FIELDS,
    DEFAULT_FACTOR,
    MODULE_FLOORS_ATTR,
    extract_snapshots_from_log,
    gpu_family_from_text,
    mean_floor,
    normalize_test_file,
)

REPO = "sgl-project/sglang"
PR_TEST_WORKFLOW = "pr-test.yml"
NIGHTLY_WORKFLOW = "nightly-test-nvidia.yml"

TEST_START_RE = re.compile(
    r"python3\s+(?:\S*?/)?(?P<path>test/(?:registered|manual)/\S+\.py)"
)
FILENAME_END_RE = re.compile(
    r"filename=['\"]?(?:\S+/)?(?P<path>test/(?:registered|manual)/\S+\.py)"
)
SUITE_FROM_RUN_SUITE_RE = re.compile(
    r"run_suite\.py\b[^\n]*?--suite\s+(?P<suite>[^\s\\]+)"
)
SUITE_FROM_JOB_RE = re.compile(
    r"(?P<suite>"
    r"(?:base|extra|stage)-[a-z]-[a-z0-9-]+"
    r"|nightly-[a-z0-9-]+"
    r"|per-commit-[a-z0-9-]+"
    r")"
)

# Marker comments around the injected block.
_BEGIN = "# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---"
_END = "# --- MEMORY_CAPACITY_FLOORS end ---"


@dataclass
class LaunchObservation:
    suite: str
    test_file: str
    run_id: str
    job_id: str
    launch_idx: int
    metrics: Dict[str, float]
    gpu_family: str  # e.g. h200, b200, h100, 5090


@dataclass
class Aggregate:
    samples: List[Dict[str, float]] = field(default_factory=list)

    def add(self, metrics: Dict[str, float]) -> None:
        self.samples.append(dict(metrics))

    def floor(self, factor: float) -> Dict[str, float]:
        by_field: Dict[str, List[float]] = defaultdict(list)
        for s in self.samples:
            for k, v in s.items():
                if k in CAPACITY_FIELDS:
                    by_field[k].append(float(v))
        out: Dict[str, float] = {}
        for k, vals in by_field.items():
            fl = mean_floor(vals, factor=factor)
            if k.endswith("_gb"):
                out[k] = round(fl, 4)
            else:
                out[k] = int(fl)
        return out


def _run(cmd: List[str], *, check: bool = True) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(
            f"Command failed ({r.returncode}): {' '.join(cmd)}\n{r.stderr}"
        )
    return r.stdout


def list_recent_runs(
    workflow: str,
    *,
    event: Optional[str] = None,
    limit: int = 5,
    branch: str = "main",
) -> List[dict]:
    q = f"repos/{REPO}/actions/workflows/{workflow}/runs?per_page={limit}&branch={branch}"
    if event:
        q += f"&event={event}"
    raw = _run(["gh", "api", q])
    data = json.loads(raw)
    return [r for r in data.get("workflow_runs", []) if r.get("status") == "completed"]


def list_jobs(run_id: int | str) -> List[dict]:
    jobs: List[dict] = []
    page = 1
    while True:
        raw = _run(
            [
                "gh",
                "api",
                f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100&page={page}",
            ]
        )
        data = json.loads(raw)
        batch = data.get("jobs", [])
        if not batch:
            break
        jobs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return jobs


def download_job_log(job_id: int | str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["gh", "run", "view", f"--job={job_id}", "--log"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not r.stdout:
        r2 = subprocess.run(
            ["gh", "api", f"repos/{REPO}/actions/jobs/{job_id}/logs"],
            capture_output=True,
            text=True,
        )
        if r2.returncode != 0 or not r2.stdout:
            return False
        dest.write_text(r2.stdout, errors="replace")
        return True
    dest.write_text(r.stdout, errors="replace")
    return True


def suite_from_run_suite_log(text: str) -> Optional[str]:
    m = SUITE_FROM_RUN_SUITE_RE.search(text)
    return m.group("suite").strip() if m else None


def suite_from_job_name(job_name: str) -> str:
    parts = [p.strip() for p in job_name.split(" / ")]
    for part in reversed(parts):
        head = re.sub(r"\s*\(\d+\)\s*$", "", part).strip()
        m = SUITE_FROM_JOB_RE.search(head)
        if m:
            return m.group("suite")
    head = re.sub(r"\s*\(\d+\)\s*$", "", parts[0] if parts else job_name).strip()
    return head or "_unknown_suite"


def parse_job_log(
    text: str,
    *,
    suite: str,
    run_id: str,
    job_id: str,
    job_name: str = "",
) -> List[LaunchObservation]:
    suite = suite_from_run_suite_log(text) or suite
    # Job display name often has the chip when the suite does not
    # (e.g. nightly-8-gpu-common on 8-gpu-h200 vs 8-gpu-b200).
    gpu_family = (
        gpu_family_from_text(job_name) or gpu_family_from_text(suite) or "unknown"
    )
    current_test: Optional[str] = None
    buffers: Dict[str, List[str]] = defaultdict(list)
    test_order: List[str] = []

    for line in text.splitlines():
        m = TEST_START_RE.search(line)
        if m:
            current_test = normalize_test_file(m.group("path"))
            if current_test not in buffers:
                test_order.append(current_test)
            continue
        if current_test is None:
            m2 = FILENAME_END_RE.search(line)
            if m2:
                current_test = normalize_test_file(m2.group("path"))
                if current_test not in buffers:
                    test_order.append(current_test)
        if current_test is not None:
            buffers[current_test].append(line)

    observations: List[LaunchObservation] = []
    for test_file in test_order:
        snaps = extract_snapshots_from_log("\n".join(buffers[test_file]))
        for idx, snap in enumerate(snaps):
            if not snap:
                continue
            observations.append(
                LaunchObservation(
                    suite=suite,
                    test_file=test_file,
                    run_id=str(run_id),
                    job_id=str(job_id),
                    launch_idx=idx,
                    metrics=snap,
                    gpu_family=gpu_family,
                )
            )
    return observations


def collect_from_log_dir(log_dir: Path) -> List[LaunchObservation]:
    obs: List[LaunchObservation] = []
    for path in sorted(log_dir.rglob("*.txt")):
        text = path.read_text(errors="replace")
        first = text.splitlines()[0] if text else ""
        job_name = first.split("\t")[0] if "\t" in first else path.stem
        suite = suite_from_job_name(job_name)
        obs.extend(
            parse_job_log(
                text,
                suite=suite,
                run_id="local",
                job_id=path.stem,
                job_name=job_name,
            )
        )
    return obs


def collect_from_runs(
    run_ids: Sequence[str],
    *,
    cache_dir: Path,
    max_jobs_per_run: Optional[int] = None,
    job_name_filter: Optional[str] = None,
) -> List[LaunchObservation]:
    obs: List[LaunchObservation] = []
    for run_id in run_ids:
        print(f"Listing jobs for run {run_id}...", flush=True)
        jobs = list_jobs(run_id)
        gpu_jobs = [
            j
            for j in jobs
            if "gpu" in j.get("name", "").lower()
            or "nightly" in j.get("name", "").lower()
        ]
        if job_name_filter:
            gpu_jobs = [j for j in gpu_jobs if job_name_filter in j.get("name", "")]
        if max_jobs_per_run is not None:
            gpu_jobs = gpu_jobs[:max_jobs_per_run]
        print(f"  {len(gpu_jobs)} jobs to download", flush=True)
        for j in gpu_jobs:
            jid = j["id"]
            name = j.get("name", "")
            suite = suite_from_job_name(name)
            dest = cache_dir / f"run_{run_id}" / f"job_{jid}.txt"
            print(f"  downloading job {jid} ({name})...", flush=True)
            if not download_job_log(jid, dest):
                print(f"    FAILED to download job {jid}", flush=True)
                continue
            text = dest.read_text(errors="replace")
            n_before = len(obs)
            obs.extend(
                parse_job_log(
                    text,
                    suite=suite,
                    run_id=str(run_id),
                    job_id=str(jid),
                    job_name=name,
                )
            )
            print(f"    +{len(obs) - n_before} launch snapshots", flush=True)
    return obs


def resolve_default_run_ids(limit: int) -> List[str]:
    run_ids: List[str] = []
    print("Fetching recent scheduled pr-test runs...", flush=True)
    for r in list_recent_runs(PR_TEST_WORKFLOW, event="schedule", limit=limit):
        run_ids.append(str(r["id"]))
        print(f"  pr-test {r['id']} {r.get('created_at')} {r.get('conclusion')}")
    print("Fetching recent nightly-test-nvidia runs...", flush=True)
    for r in list_recent_runs(NIGHTLY_WORKFLOW, limit=limit):
        if r.get("head_branch") and r["head_branch"] != "main":
            continue
        run_ids.append(str(r["id"]))
        print(f"  nightly {r['id']} {r.get('created_at')} {r.get('conclusion')}")
    return run_ids


def group_by_file_gpu(
    obs: Sequence[LaunchObservation],
) -> Dict[Tuple[str, str], Dict[int, Aggregate]]:
    """(test_file, gpu_family) -> launch_idx -> Aggregate."""
    grouped: Dict[Tuple[str, str], Dict[int, Aggregate]] = defaultdict(
        lambda: defaultdict(Aggregate)
    )
    for o in obs:
        if o.gpu_family == "unknown":
            continue
        grouped[(o.test_file, o.gpu_family)][o.launch_idx].add(o.metrics)
    return grouped


def floors_for_group(
    by_idx: Dict[int, Aggregate], *, factor: float
) -> Tuple[List[Dict[str, float]], List[int]]:
    launches: List[Dict[str, float]] = []
    sample_counts: List[int] = []
    for idx in sorted(by_idx.keys()):
        agg = by_idx[idx]
        fl = agg.floor(factor=factor)
        if not fl:
            continue
        launches.append(fl)
        sample_counts.append(len(agg.samples))
    return launches, sample_counts


def _format_launch_list(launches: List[Dict[str, float]], *, indent: str) -> List[str]:
    lines = [f"{indent}["]
    inner = indent + "    "
    for launch in launches:
        items = ", ".join(
            f'"{k}": {launch[k]!r}' for k in CAPACITY_FIELDS if k in launch
        )
        lines.append(f"{inner}{{{items}}},")
    lines.append(f"{indent}]")
    return lines


def format_floors_block(
    by_gpu: Dict[str, Tuple[List[Dict[str, float]], List[int]]],
) -> str:
    """Format list (single GPU) or dict (multi-GPU) MEMORY_CAPACITY_FLOORS."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [_BEGIN]
    if len(by_gpu) == 1:
        gpu, (launches, counts) = next(iter(by_gpu.items()))
        lines.append(f"# gpu={gpu} samples={counts} updated={today}")
        lines.append(f"{MODULE_FLOORS_ATTR} = [")
        for launch in launches:
            items = ", ".join(
                f'"{k}": {launch[k]!r}' for k in CAPACITY_FIELDS if k in launch
            )
            lines.append(f"    {{{items}}},")
        lines.append("]")
    else:
        meta = ", ".join(f"{g}:samples={by_gpu[g][1]}" for g in sorted(by_gpu.keys()))
        lines.append(f"# multi-gpu floors; {meta} updated={today}")
        lines.append(f"{MODULE_FLOORS_ATTR} = {{")
        for gpu in sorted(by_gpu.keys()):
            launches, counts = by_gpu[gpu]
            lines.append(f"    # samples={counts}")
            lines.append(f'    "{gpu}": [')
            for launch in launches:
                items = ", ".join(
                    f'"{k}": {launch[k]!r}' for k in CAPACITY_FIELDS if k in launch
                )
                lines.append(f"        {{{items}}},")
            lines.append("    ],")
        lines.append("}")
    lines.append(_END)
    return "\n".join(lines) + "\n"


def _find_injection_index(src: str) -> int:
    """Insert after the module docstring, imports, and register_*_ci calls."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0
    last_end = 0
    for i, node in enumerate(tree.body):
        # Module docstring
        if (
            i == 0
            and isinstance(node, ast.Expr)
            and isinstance(getattr(node, "value", None), ast.Constant)
            and isinstance(node.value.value, str)
        ):
            last_end = getattr(node, "end_lineno", node.lineno) or node.lineno
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_end = getattr(node, "end_lineno", node.lineno) or node.lineno
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            # module-level register_cuda_ci(...) etc.
            last_end = getattr(node, "end_lineno", node.lineno) or node.lineno
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            # Keep going past simple constants that often sit near imports.
            last_end = getattr(node, "end_lineno", node.lineno) or node.lineno
            continue
        break
    if last_end <= 0:
        return 0
    lines = src.splitlines(keepends=True)
    return sum(len(lines[i]) for i in range(min(last_end, len(lines))))


def _strip_existing_floors_block(src: str) -> str:
    """Remove a previous auto-generated MEMORY_CAPACITY_FLOORS block."""
    if _BEGIN not in src or _END not in src:
        return src
    pre, rest = src.split(_BEGIN, 1)
    _, post = rest.split(_END, 1)
    # Trim blank lines left at the join so re-injection is clean.
    pre = pre.rstrip("\n")
    post = post.lstrip("\n")
    if pre and post:
        return pre + "\n\n" + post
    return pre + post


def inject_floors_into_source(
    src: str,
    by_gpu: Dict[str, Tuple[List[Dict[str, float]], List[int]]],
) -> str:
    block = format_floors_block(by_gpu)
    # Always strip + re-inject so a previous wrong position is corrected.
    src = _strip_existing_floors_block(src)
    idx = _find_injection_index(src)
    pre, post = src[:idx], src[idx:]
    if pre and not pre.endswith("\n"):
        pre += "\n"
    if post and not post.startswith("\n"):
        post = "\n" + post
    return pre + "\n" + block + post


def write_floors_to_files(
    file_floors: Dict[str, Dict[str, Tuple[List[Dict[str, float]], List[int]]]],
    *,
    dry_run: bool,
) -> int:
    updated = 0
    for test_file, by_gpu in sorted(file_floors.items()):
        path = REPO_ROOT / test_file
        if not path.is_file():
            print(f"  SKIP missing {test_file}", flush=True)
            continue
        old = path.read_text(encoding="utf-8")
        new = inject_floors_into_source(old, by_gpu)
        gpus = ",".join(sorted(by_gpu.keys()))
        n_launch = max(len(v[0]) for v in by_gpu.values())
        if new == old:
            print(
                f"  unchanged {test_file} gpus=[{gpus}] launches={n_launch}",
                flush=True,
            )
            continue
        print(
            f"  {'would update' if dry_run else 'update'} {test_file} "
            f"gpus=[{gpus}] launches={n_launch}",
            flush=True,
        )
        if not dry_run:
            path.write_text(new, encoding="utf-8")
        updated += 1
    return updated


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-id", action="append", default=[])
    p.add_argument("--limit-runs", type=int, default=3)
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "sglang_memory_threshold_logs",
    )
    p.add_argument("--factor", type=float, default=DEFAULT_FACTOR)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-jobs-per-run", type=int, default=None)
    p.add_argument("--job-name-filter", type=str, default=None)
    args = p.parse_args(argv)

    if args.log_dir:
        observations = collect_from_log_dir(args.log_dir)
    else:
        import shutil

        if not shutil.which("gh"):
            print(
                "Error: the 'gh' (GitHub) CLI is required but was not found in PATH.\n"
                "Install it and run 'gh auth login' to authenticate.",
                file=sys.stderr,
            )
            return 1
        run_ids = args.run_id or resolve_default_run_ids(args.limit_runs)
        if not run_ids:
            print("No runs found.", file=sys.stderr)
            return 1
        observations = collect_from_runs(
            run_ids,
            cache_dir=args.cache_dir,
            max_jobs_per_run=args.max_jobs_per_run,
            job_name_filter=args.job_name_filter,
        )

    print(f"Collected {len(observations)} launch observations", flush=True)
    if not observations:
        print("Nothing to write.", file=sys.stderr)
        return 1

    grouped = group_by_file_gpu(observations)
    # test_file -> gpu_family -> (launches, counts)
    file_floors: Dict[str, Dict[str, Tuple[List[Dict[str, float]], List[int]]]] = (
        defaultdict(dict)
    )
    for (test_file, gpu), by_idx in grouped.items():
        launches, counts = floors_for_group(by_idx, factor=args.factor)
        if launches:
            file_floors[test_file][gpu] = (launches, counts)

    multi = sum(1 for v in file_floors.values() if len(v) > 1)
    print(
        f"Floors for {len(file_floors)} test files " f"({multi} multi-GPU)",
        flush=True,
    )
    n = write_floors_to_files(file_floors, dry_run=args.dry_run)
    print(
        f"{'Would update' if args.dry_run else 'Updated'} {n} file(s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
