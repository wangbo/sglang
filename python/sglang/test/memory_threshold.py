"""E2E memory-capacity floors for CI server-launching tests.

Each test module (or class) declares floors explicitly::

    # Single-hardware (any GPU that runs this suite):
    MEMORY_CAPACITY_FLOORS = [
        {"token_capacity": 52358, "kv_cache_gb": 6.39},
    ]

    # Multi-hardware (same test on H200 and B200, etc.):
    MEMORY_CAPACITY_FLOORS = {
        "h200": [{"token_capacity": 11111601, "kv_cache_gb": 11.92}],
        "b200": [{"token_capacity": 15000000, "kv_cache_gb": 15.0}],
    }

    class TestFoo(CustomTestCase):
        # Optional per-class override (else the module value is used):
        # memory_capacity_floors = [...]

After ``popen_launch_server`` (or PD health) becomes ready, the harness
``GET /server_info`` and asserts observed capacity >= the next unused floor
for the active test class / module and current GPU family.

Offline: ``scripts/ci/utils/update_memory_thresholds.py`` mines scheduled
PR-test / nightly logs and rewrites ``MEMORY_CAPACITY_FLOORS`` in each file.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import sys
import threading
import types
import unittest
from typing import Any, Dict, List, Optional, Sequence, Union

import requests

logger = logging.getLogger(__name__)

DEFAULT_FACTOR = 0.99

# Capacity fields: higher is better (more tokens / larger usable pools).
CAPACITY_FIELDS = (
    "token_capacity",  # max_total_num_tokens / #tokens
    "kv_cache_gb",  # allocated KV pool GB
    "swa_size",
    "full_size",
    "swa_mem_gb",
    "mamba_cache_size",
    "mamba_conv_gb",
    "mamba_ssm_gb",
    "dsv4_full",
    "dsv4_swa",
    "dsv4_c4",
    "dsv4_c128",
    "dsv4_c4_state",
    "dsv4_c128_state",
)

# Stable GPU family keys used in MEMORY_CAPACITY_FLOORS dict form.
# Longer / more specific tokens first for matching.
GPU_FAMILY_TOKENS = (
    "gb300",
    "gb200",
    "b200",
    "h200",
    "h100",
    "h20",
    "a100",
    "5090",
    "4090",
    "l40s",
    "l40",
)

# Module attribute name written by the update script / hand-authored tests.
MODULE_FLOORS_ATTR = "MEMORY_CAPACITY_FLOORS"
# Optional per-class override.
CLASS_FLOORS_ATTR = "memory_capacity_floors"
# Per-owner launch counter attribute (mutated at runtime).
_FLOOR_IDX_ATTR = "_memory_capacity_floor_idx"

# ---- log line parsers (shared with the update script) ----

KV_RE = re.compile(
    r"KV Cache is allocated\.\s*dtype:\s*(?P<dtype>\S+),\s*#tokens:\s*(?P<tokens>\d+),\s*"
    r"(?:KV size:\s*(?P<kv_size>[\d.]+)\s*GB|"
    r"K size:\s*(?P<k_size>[\d.]+)\s*GB,\s*V size:\s*(?P<v_size>[\d.]+)\s*GB)"
)
SWA_RE = re.compile(
    r"SWAKVPool mem usage:\s*(?P<mem>[\d.]+)\s*GB,\s*"
    r"swa size:\s*(?P<swa>\d+),\s*full size:\s*(?P<full>\d+)"
)
MAMBA_RE = re.compile(
    r"Mamba Cache is allocated\.\s*max_mamba_cache_size:\s*(?P<mamba>\d+),\s*"
    r"conv_state size:\s*(?P<conv>[\d.]+)\s*GB,?\s*"
    r"ssm_state size:\s*(?P<ssm>[\d.]+)\s*GB"
)
DSV4_RE = re.compile(
    r"DSV4 pool sizes:\s*full=(?P<full>\d+),\s*swa=(?P<swa>\d+),\s*"
    r"c4=(?P<c4>\d+),\s*c128=(?P<c128>\d+),\s*"
    r"c4_state=(?P<c4_state>\d+),\s*c128_state=(?P<c128_state>\d+)"
)

_lock = threading.Lock()

FloorOwner = Union[type, types.ModuleType]
FloorDict = Dict[str, float]


def parse_memory_log_line(line: str) -> Optional[FloorDict]:
    """Parse one engine log line into a partial capacity snapshot."""
    m = KV_RE.search(line)
    if m:
        tokens = int(m.group("tokens"))
        if m.group("kv_size") is not None:
            kv_gb = float(m.group("kv_size"))
        else:
            kv_gb = float(m.group("k_size")) + float(m.group("v_size"))
        return {"token_capacity": tokens, "kv_cache_gb": kv_gb}

    m = SWA_RE.search(line)
    if m:
        return {
            "swa_mem_gb": float(m.group("mem")),
            "swa_size": int(m.group("swa")),
            "full_size": int(m.group("full")),
            "token_capacity": int(m.group("full")),
        }

    m = MAMBA_RE.search(line)
    if m:
        return {
            "mamba_cache_size": int(m.group("mamba")),
            "mamba_conv_gb": float(m.group("conv")),
            "mamba_ssm_gb": float(m.group("ssm")),
        }

    m = DSV4_RE.search(line)
    if m:
        return {
            "dsv4_full": int(m.group("full")),
            "dsv4_swa": int(m.group("swa")),
            "dsv4_c4": int(m.group("c4")),
            "dsv4_c128": int(m.group("c128")),
            "dsv4_c4_state": int(m.group("c4_state")),
            "dsv4_c128_state": int(m.group("c128_state")),
            "token_capacity": int(m.group("full")),
        }

    return None


def _fingerprint(snap: FloorDict) -> tuple:
    return tuple(sorted((k, snap[k]) for k in CAPACITY_FIELDS if k in snap))


def extract_snapshots_from_log(text: str) -> List[FloorDict]:
    """Extract ordered capacity snapshots from engine log text.

    Multi-TP ranks emit identical allocation lines; consecutive identical
    fingerprints are collapsed. Related lines from a single server start
    (Mamba + KV, SWA sub-pools, EAGLE target+draft) are merged so each
    snapshot approximates one ``GET /server_info`` sample.
    """
    raw: List[FloorDict] = []
    for line in text.splitlines():
        snap = parse_memory_log_line(line)
        if snap is None:
            continue
        if raw and _fingerprint(snap) == _fingerprint(raw[-1]):
            continue  # TP duplicate
        if raw and _can_merge(raw[-1], snap):
            raw[-1] = {**raw[-1], **snap}
        else:
            raw.append(dict(snap))
    return _collapse_to_server_launches(raw)


def _can_merge(a: FloorDict, b: FloorDict) -> bool:
    if _is_kv_only(a) and _is_kv_only(b):
        return False
    for k in b:
        if k in a and a[k] != b[k]:
            if k in ("token_capacity", "kv_cache_gb"):
                continue
            return False
    return True


def _is_kv_only(snap: FloorDict) -> bool:
    return set(snap.keys()).issubset({"token_capacity", "kv_cache_gb"})


def _collapse_to_server_launches(snaps: List[FloorDict]) -> List[FloorDict]:
    """Collapse log lines into one snapshot per ``popen_launch_server``.

    * Hybrid SWA: two sub-pool KV lines + ``SWAKVPool`` summary.
    * Speculative (EAGLE): target + draft pure-KV (same token_capacity);
      keep the larger kv_cache_gb (target; what /server_info reports).
    """
    if not snaps:
        return snaps
    out: List[FloorDict] = []
    for snap in snaps:
        if "swa_size" in snap or "full_size" in snap:
            swa = snap.get("swa_size")
            full = snap.get("full_size") or snap.get("token_capacity")
            kept: List[FloorDict] = []
            for prev in out:
                if not _is_kv_only(prev):
                    kept.append(prev)
                    continue
                tc = prev.get("token_capacity")
                if tc is not None and tc in (swa, full):
                    if "kv_cache_gb" in prev:
                        snap["kv_cache_gb"] = max(
                            float(snap.get("kv_cache_gb", 0.0)),
                            float(prev["kv_cache_gb"]),
                        )
                    continue
                kept.append(prev)
            out = kept
            out.append(snap)
            continue

        if (
            out
            and _is_kv_only(out[-1])
            and _is_kv_only(snap)
            and out[-1].get("token_capacity") == snap.get("token_capacity")
            and out[-1].get("token_capacity") is not None
        ):
            prev = out[-1]
            if float(snap.get("kv_cache_gb", 0.0)) > float(
                prev.get("kv_cache_gb", 0.0)
            ):
                out[-1] = dict(snap)
            continue

        out.append(snap)
    return out


def snapshot_from_server_info(info: Dict[str, Any]) -> FloorDict:
    """Build a capacity snapshot from a ``/server_info`` JSON response."""
    snap: FloorDict = {}

    if "max_total_num_tokens" in info and info["max_total_num_tokens"] is not None:
        snap["token_capacity"] = int(info["max_total_num_tokens"])

    mem = None
    internal = info.get("internal_states")
    if isinstance(internal, list) and internal and isinstance(internal[0], dict):
        mem = internal[0].get("memory_usage")
    if not isinstance(mem, dict):
        mem = info.get("memory_usage")
    if not isinstance(mem, dict):
        mem = {}

    if "token_capacity" in mem and mem["token_capacity"] is not None:
        snap["token_capacity"] = int(mem["token_capacity"])
    if "kvcache" in mem and mem["kvcache"] is not None:
        snap["kv_cache_gb"] = float(mem["kvcache"])

    int_fields = (
        "swa_size",
        "full_size",
        "mamba_cache_size",
        "dsv4_full",
        "dsv4_swa",
        "dsv4_c4",
        "dsv4_c128",
        "dsv4_c4_state",
        "dsv4_c128_state",
    )
    float_fields = ("swa_mem_gb", "mamba_conv_gb", "mamba_ssm_gb")
    for field in int_fields:
        if field in mem and mem[field] is not None:
            snap[field] = int(mem[field])
    for field in float_fields:
        if field in mem and mem[field] is not None:
            snap[field] = float(mem[field])

    return snap


def fetch_server_memory_snapshot(
    base_url: str,
    *,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> FloorDict:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.get(
        f"{base_url.rstrip('/')}/server_info",
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    return snapshot_from_server_info(resp.json())


def normalize_test_file(path: str) -> str:
    """Strip absolute CI checkout prefixes down to ``test/...``."""
    path = path.replace("\\", "/")
    for marker in (
        "/sglang/test/",
        "test/registered/",
        "test/manual/",
        "test/",
    ):
        idx = path.find(marker)
        if idx >= 0:
            if marker.startswith("/sglang/"):
                return path[idx + len("/sglang/") :]
            return path[idx:]
    return path.lstrip("./")


def mean_floor(values: Sequence[float], factor: float = DEFAULT_FACTOR) -> float:
    if not values:
        raise ValueError("mean_floor requires non-empty values")
    return sum(values) / len(values) * factor


def check_snapshot_against_floor(
    observed: FloorDict,
    floor: FloorDict,
    *,
    label: str = "",
) -> List[str]:
    """Return failure messages (empty if OK)."""
    failures: List[str] = []
    for field in CAPACITY_FIELDS:
        if field not in floor:
            continue
        if field not in observed:
            logger.warning(
                "Memory floor %s: has %s=%.4g but server did not report it; "
                "skipping field",
                label or "?",
                field,
                floor[field],
            )
            continue
        obs = float(observed[field])
        thr = float(floor[field])
        if obs < thr:
            failures.append(
                f"{field}: observed={obs:g} < floor={thr:g}"
                + (f" ({label})" if label else "")
            )
    return failures


def gpu_family_from_text(text: str) -> Optional[str]:
    """Map suite / job / device name text to a stable GPU family key."""
    s = text.lower().replace("_", "-")
    for key in GPU_FAMILY_TOKENS:
        if key in s:
            return key
    # Runner-config / suite shorthands without the chip in the name.
    if "1-gpu-small" in s:
        return "5090"
    if "1-gpu-large" in s or "2-gpu-large" in s:
        return "h100"
    if re.search(r"(^|[^a-z])4-gpu-h100|deepep-4-gpu-h100", s):
        return "h100"
    if re.search(r"(^|[^a-z])4-gpu-b200|deepep-4-gpu-b200", s):
        return "b200"
    if re.search(r"8-gpu-h200|deepep-8-gpu-h200", s):
        return "h200"
    if re.search(r"8-gpu-b200", s):
        return "b200"
    if re.search(r"8-gpu-h20", s):
        return "h20"
    if "gb300" in s or "gb200" in s:
        return "gb300" if "gb300" in s else "gb200"
    return None


def detect_gpu_family() -> Optional[str]:
    """Runtime GPU family for selecting multi-hardware floors."""
    env = os.environ.get("SGLANG_MEMORY_FLOOR_GPU", "").strip().lower()
    if env:
        return env
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_properties(0).name
    except Exception:
        return None
    return gpu_family_from_text(name)


def _raw_floors_from_owner(owner: FloorOwner) -> Any:
    if isinstance(owner, type):
        class_floors = getattr(owner, CLASS_FLOORS_ATTR, None)
        if class_floors is not None:
            return class_floors
        mod = sys.modules.get(owner.__module__)
        if mod is not None:
            return getattr(mod, MODULE_FLOORS_ATTR, None)
        return None
    return getattr(owner, MODULE_FLOORS_ATTR, None)


def resolve_launch_floors(
    floors_spec: Any, *, gpu_family: Optional[str] = None
) -> Optional[List[FloorDict]]:
    """Normalize MEMORY_CAPACITY_FLOORS list or per-GPU dict to a launch list.

    * ``list`` — used on every GPU (single-runner tests).
    * ``dict`` — keyed by GPU family (``h200``, ``b200``, …); only the entry
      matching the current GPU is used. Missing key → no check (do not fall
      back to another GPU's floors).
    """
    if floors_spec is None:
        return None
    if isinstance(floors_spec, list):
        return list(floors_spec) if floors_spec else None
    if isinstance(floors_spec, dict):
        if not floors_spec:
            return None
        family = gpu_family if gpu_family is not None else detect_gpu_family()
        if family is None:
            logger.warning(
                "MEMORY_CAPACITY_FLOORS is a per-GPU dict %s but GPU family "
                "could not be detected; skipping memory floor check",
                list(floors_spec.keys()),
            )
            return None
        if family not in floors_spec:
            logger.warning(
                "MEMORY_CAPACITY_FLOORS has keys %s but no entry for "
                "gpu_family=%s; skipping memory floor check",
                list(floors_spec.keys()),
                family,
            )
            return None
        launches = floors_spec[family]
        return list(launches) if launches else None
    logger.warning(
        "MEMORY_CAPACITY_FLOORS has unsupported type %s; expected list or dict",
        type(floors_spec).__name__,
    )
    return None


def _floors_from_owner(owner: FloorOwner) -> Optional[List[FloorDict]]:
    return resolve_launch_floors(_raw_floors_from_owner(owner))


def _owner_label(owner: FloorOwner) -> str:
    if isinstance(owner, type):
        return f"{owner.__module__}.{owner.__qualname__}"
    return getattr(owner, "__name__", repr(owner))


def claim_next_memory_floor(owner: FloorOwner) -> Optional[tuple[FloorDict, int]]:
    """Return ``(floor, launch_idx)`` for the next server launch, or None."""
    floors = _floors_from_owner(owner)
    if not floors:
        return None
    with _lock:
        # Counter lives on the owner that actually holds the floors (class if
        # class override, else the defining module) so multi-class files that
        # share module floors share one sequence.
        counter_owner: FloorOwner = owner
        if isinstance(owner, type):
            class_floors = getattr(owner, CLASS_FLOORS_ATTR, None)
            if class_floors is None:
                mod = sys.modules.get(owner.__module__)
                if (
                    mod is not None
                    and getattr(mod, MODULE_FLOORS_ATTR, None) is not None
                ):
                    counter_owner = mod
        idx = int(getattr(counter_owner, _FLOOR_IDX_ATTR, 0))
        setattr(counter_owner, _FLOOR_IDX_ATTR, idx + 1)
    if idx >= len(floors):
        logger.info(
            "Memory floors %s: launch[%d] beyond declared %d; skipping",
            _owner_label(owner),
            idx,
            len(floors),
        )
        return None
    return floors[idx], idx


def find_active_test_owner() -> Optional[FloorOwner]:
    """Locate the unittest class (or its module) declaring floors.

    Walks the stack for a ``cls`` local that is a ``TestCase`` subclass —
    the usual pattern in ``setUpClass`` / fixture launch helpers.
    """
    for frame_info in inspect.stack(context=0):
        loc = frame_info.frame.f_locals
        cls = loc.get("cls")
        if not isinstance(cls, type):
            continue
        try:
            if not issubclass(cls, unittest.TestCase):
                continue
        except TypeError:
            continue
        if _raw_floors_from_owner(cls) is not None:
            return cls
    # Fallback: __main__ module floors (``python path/to/test.py``).
    main = sys.modules.get("__main__")
    if main is not None and getattr(main, MODULE_FLOORS_ATTR, None) is not None:
        return main
    return None


def memory_threshold_check_enabled() -> bool:
    """On in NVIDIA CI by default; force with SGLANG_CHECK_MEMORY_THRESHOLDS=1/0.

    Floors are mined from NVIDIA scheduled/nightly logs, so skip on AMD CI
    (SGLANG_IS_IN_CI_AMD) where free GPU memory / capacity differ.
    """
    flag = os.environ.get("SGLANG_CHECK_MEMORY_THRESHOLDS", "").lower()
    if flag in ("0", "false", "no", "off"):
        return False
    # Explicit force-on still runs even on AMD (for local experiments).
    if flag in ("1", "true", "yes", "on"):
        return True
    if os.environ.get("SGLANG_IS_IN_CI_AMD", "").lower() in ("1", "true", "yes"):
        return False
    return os.environ.get("SGLANG_IS_IN_CI", "").lower() in ("1", "true", "yes")


def maybe_check_server_memory(
    base_url: str,
    *,
    api_key: Optional[str] = None,
    floor: Optional[FloorDict] = None,
    owner: Optional[FloorOwner] = None,
) -> None:
    """Assert capacity against an explicit floor or the next declared floor.

    No-op when disabled, when no floor is available, or when /server_info
    cannot be queried. Raises ``AssertionError`` on regression.
    """
    if not memory_threshold_check_enabled():
        return

    label = ""
    launch_idx = -1
    if floor is None:
        owner = owner or find_active_test_owner()
        if owner is None:
            return
        claimed = claim_next_memory_floor(owner)
        if claimed is None:
            return
        floor, launch_idx = claimed
        label = f"{_owner_label(owner)} launch[{launch_idx}]"
    else:
        label = "explicit floor"

    try:
        observed = fetch_server_memory_snapshot(base_url, api_key=api_key)
    except Exception as e:
        logger.warning(
            "Memory floor check skipped for %s: failed to query /server_info (%s)",
            label,
            e,
        )
        return

    if not observed:
        logger.warning("Memory floor check skipped for %s: empty snapshot", label)
        return

    logger.info("Memory floor check %s: observed=%s floor=%s", label, observed, floor)
    failures = check_snapshot_against_floor(observed, floor, label=label)
    if failures:
        raise AssertionError(
            "Memory capacity regression detected:\n  "
            + "\n  ".join(failures)
            + "\nUpdate MEMORY_CAPACITY_FLOORS in the test file after an "
            "intentional optimization via "
            "scripts/ci/utils/update_memory_thresholds.py"
        )


def reset_floor_counters(*owners: FloorOwner) -> None:
    """Test helper: clear launch counters on the given owners."""
    with _lock:
        for owner in owners:
            if hasattr(owner, _FLOOR_IDX_ATTR):
                delattr(owner, _FLOOR_IDX_ATTR)
