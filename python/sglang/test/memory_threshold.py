"""E2E guard for allocated KV-related buffer size (MB).

Metric (from ``GET /server_info`` → ``memory_usage.kv_buffer_mb``):

  KV / SWA / DSA / unified token pools (``kvcache.mem_usage``)
  + Mamba / GDN-like state pools when present

Weights and CUDA-graph memory are **not** included.

Declare on the test module or class::

    MIN_KV_BUFFER_MB = 12000
    MIN_KV_BUFFER_MB = {"h200": 12000, "b200": 18000}
    MIN_KV_BUFFER_MB = [12000, 800]  # multi-launch

After ``popen_launch_server`` / PD health: assert ``kv_buffer_mb >= floor``.

Disabled on AMD CI. Opt out: ``SGLANG_CHECK_MEMORY_THRESHOLDS=0``.
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
from typing import Any, Dict, List, Optional, Union

import requests

logger = logging.getLogger(__name__)

DEFAULT_FACTOR = 0.99

MODULE_MIN_ATTR = "MIN_KV_BUFFER_MB"
CLASS_MIN_ATTR = "min_kv_buffer_mb"

_USED_ATTR = "_min_kv_buffer_mb_used"
_CHECKED_PIDS: set[int] = set()

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

_lock = threading.Lock()

FloorOwner = Union[type, types.ModuleType]


def gpu_family_from_text(text: str) -> Optional[str]:
    s = text.lower().replace("_", "-")
    for key in GPU_FAMILY_TOKENS:
        if key in s:
            return key
    if "1-gpu-small" in s:
        return "5090"
    if "1-gpu-large" in s or "2-gpu-large" in s:
        return "h100"
    if re.search(r"4-gpu-h100|deepep-4-gpu-h100", s):
        return "h100"
    if re.search(r"4-gpu-b200|deepep-4-gpu-b200", s):
        return "b200"
    if re.search(r"8-gpu-h200|deepep-8-gpu-h200", s):
        return "h200"
    if re.search(r"8-gpu-b200", s):
        return "b200"
    if re.search(r"8-gpu-h20", s):
        return "h20"
    if "gb300" in s:
        return "gb300"
    if "gb200" in s:
        return "gb200"
    return None


def detect_gpu_family() -> Optional[str]:
    env = os.environ.get("SGLANG_MEMORY_FLOOR_GPU", "").strip().lower()
    if env:
        return env
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return gpu_family_from_text(torch.cuda.get_device_properties(0).name)
    except Exception:
        return None


def kv_buffer_mb_from_server_info(info: Dict[str, Any]) -> Optional[float]:
    """Extract kv_buffer_mb from /server_info."""
    mem = None
    internal = info.get("internal_states")
    if isinstance(internal, list) and internal and isinstance(internal[0], dict):
        mem = internal[0].get("memory_usage")
    if not isinstance(mem, dict):
        mem = info.get("memory_usage")
    if not isinstance(mem, dict):
        return None
    if mem.get("kv_buffer_mb") is not None:
        return float(mem["kv_buffer_mb"])
    # Fallback for older servers: kvcache (+ mamba) in GB → MB.
    parts = []
    if mem.get("kvcache") is not None:
        parts.append(float(mem["kvcache"]))
    if mem.get("mamba") is not None:
        parts.append(float(mem["mamba"]))
    if not parts:
        return None
    return round(sum(parts) * 1024.0, 1)


def fetch_kv_buffer_mb(
    base_url: str,
    *,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> Optional[float]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.get(
        f"{base_url.rstrip('/')}/server_info",
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    return kv_buffer_mb_from_server_info(resp.json())


def _raw_min_from_owner(owner: FloorOwner) -> Any:
    if isinstance(owner, type):
        v = getattr(owner, CLASS_MIN_ATTR, None)
        if v is not None:
            return v
        mod = sys.modules.get(owner.__module__)
        if mod is not None:
            return getattr(mod, MODULE_MIN_ATTR, None)
        return None
    return getattr(owner, MODULE_MIN_ATTR, None)


def resolve_min_kv_buffer_mb(
    spec: Any, *, gpu_family: Optional[str] = None
) -> Optional[List[float]]:
    """Normalize MIN_KV_BUFFER_MB to a list of floors (one per launch)."""
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return [float(spec)]
    if isinstance(spec, list):
        return [float(x) for x in spec] if spec else None
    if isinstance(spec, dict):
        family = gpu_family if gpu_family is not None else detect_gpu_family()
        if family is None:
            logger.warning(
                "MIN_KV_BUFFER_MB is per-GPU %s but GPU family unknown; skip",
                list(spec.keys()),
            )
            return None
        if family not in spec:
            logger.warning(
                "MIN_KV_BUFFER_MB keys %s have no entry for %s; skip",
                list(spec.keys()),
                family,
            )
            return None
        return resolve_min_kv_buffer_mb(spec[family], gpu_family=family)
    logger.warning("MIN_KV_BUFFER_MB has unsupported type %s", type(spec).__name__)
    return None


def _counter_owner(owner: FloorOwner) -> FloorOwner:
    if isinstance(owner, type) and getattr(owner, CLASS_MIN_ATTR, None) is None:
        mod = sys.modules.get(owner.__module__)
        if mod is not None and _raw_min_from_owner(mod) is not None:
            return mod
    return owner


def claim_min_kv_buffer_mb(
    owner: FloorOwner, observed_mb: float
) -> Optional[tuple[float, int]]:
    """Claim the unused floor closest to ``observed_mb``."""
    floors = resolve_min_kv_buffer_mb(_raw_min_from_owner(owner))
    if not floors:
        return None
    with _lock:
        c_owner = _counter_owner(owner)
        used = set(getattr(c_owner, _USED_ATTR, set()))
        candidates = [i for i in range(len(floors)) if i not in used]
        if not candidates:
            logger.info(
                "MIN_KV_BUFFER_MB %s: all floors claimed; skip",
                _owner_label(owner),
            )
            return None
        best_i = min(candidates, key=lambda i: abs(floors[i] - observed_mb))
        used.add(best_i)
        setattr(c_owner, _USED_ATTR, used)
    return floors[best_i], best_i


def _owner_label(owner: FloorOwner) -> str:
    if isinstance(owner, type):
        return f"{owner.__module__}.{owner.__qualname__}"
    return getattr(owner, "__name__", repr(owner))


def find_active_test_owner() -> Optional[FloorOwner]:
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
        if _raw_min_from_owner(cls) is not None:
            return cls
    main = sys.modules.get("__main__")
    if main is not None and _raw_min_from_owner(main) is not None:
        return main
    return None


def memory_threshold_check_enabled() -> bool:
    flag = os.environ.get("SGLANG_CHECK_MEMORY_THRESHOLDS", "").lower()
    if flag in ("0", "false", "no", "off"):
        return False
    if flag in ("1", "true", "yes", "on"):
        return True
    if os.environ.get("SGLANG_IS_IN_CI_AMD", "").lower() in ("1", "true", "yes"):
        return False
    return os.environ.get("SGLANG_IS_IN_CI", "").lower() in ("1", "true", "yes")


def process_memory_already_checked(process: Any) -> bool:
    pid = getattr(process, "pid", None)
    if pid is None:
        return False
    with _lock:
        return int(pid) in _CHECKED_PIDS


def mark_process_memory_checked(process: Any) -> None:
    pid = getattr(process, "pid", None)
    if pid is not None:
        with _lock:
            _CHECKED_PIDS.add(int(pid))


def maybe_check_server_memory(
    base_url: str,
    *,
    api_key: Optional[str] = None,
    process: Any = None,
    owner: Optional[FloorOwner] = None,
) -> None:
    """Assert ``memory_usage.kv_buffer_mb`` >= declared ``MIN_KV_BUFFER_MB``."""
    if not memory_threshold_check_enabled():
        return
    if process is not None and process_memory_already_checked(process):
        return

    owner = owner or find_active_test_owner()
    if owner is None or _raw_min_from_owner(owner) is None:
        return

    try:
        observed = fetch_kv_buffer_mb(base_url, api_key=api_key)
    except Exception as e:
        logger.warning("KV buffer floor check skipped: /server_info failed (%s)", e)
        return
    if observed is None:
        logger.warning("KV buffer floor check skipped: server_info has no kv_buffer_mb")
        return

    claimed = claim_min_kv_buffer_mb(owner, observed)
    if claimed is None:
        return
    floor_mb, idx = claimed
    label = f"{_owner_label(owner)} floor[{idx}]"
    logger.info(
        "KV buffer floor check %s: kv_buffer_mb observed=%.1f floor=%.1f",
        label,
        observed,
        floor_mb,
    )
    if observed < floor_mb:
        raise AssertionError(
            f"KV buffer capacity regression ({label}): "
            f"kv_buffer_mb observed={observed:g} < floor={floor_mb:g}. "
            f"Update MIN_KV_BUFFER_MB after an intentional change via "
            f"scripts/ci/utils/update_memory_thresholds.py"
        )
    if process is not None:
        mark_process_memory_checked(process)


def reset_floor_counters(*owners: FloorOwner) -> None:
    with _lock:
        for owner in owners:
            if hasattr(owner, _USED_ATTR):
                delattr(owner, _USED_ATTR)
        _CHECKED_PIDS.clear()


def mean_floor(values: list[float], factor: float = DEFAULT_FACTOR) -> float:
    if not values:
        raise ValueError("empty values")
    return sum(values) / len(values) * factor
