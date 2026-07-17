"""E2E KV-buffer floor, same style as accuracy thresholds.

Metric: ``GET /server_info`` → ``memory_usage.kv_buffer_mb``
  = allocated token KV (incl. SWA / DSA / unified) + Mamba/GDN state.
  Weights and CUDA graphs are not included.

Declare on the test class (preferred) or module::

    min_kv_buffer_mb = 12000
    # hardware-dependent when the same test runs on multiple GPU families:
    min_kv_buffer_mb = {"h200": 12000, "b200": 18000}

After ``popen_launch_server`` / PD worker health: assert ``kv_buffer_mb >= threshold``.
No declaration → no check. Missing GPU key → skip.

Enabled in CI (not AMD). Opt out: ``SGLANG_CHECK_MEMORY_THRESHOLDS=0``.
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
from typing import Any, Dict, Optional, Union

import requests

logger = logging.getLogger(__name__)

MODULE_ATTR = "MIN_KV_BUFFER_MB"
CLASS_ATTR = "min_kv_buffer_mb"

_CHECKED_PIDS: set[int] = set()
_lock = threading.Lock()

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
    # Older servers: sum kvcache + mamba (GB → MB); ignore weight/graph.
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


def _raw_threshold(owner: FloorOwner) -> Any:
    if isinstance(owner, type):
        v = getattr(owner, CLASS_ATTR, None)
        if v is not None:
            return v
        mod = sys.modules.get(owner.__module__)
        return getattr(mod, MODULE_ATTR, None) if mod is not None else None
    return getattr(owner, MODULE_ATTR, None)


def resolve_threshold(
    spec: Any, *, gpu_family: Optional[str] = None
) -> Optional[float]:
    """Return a single MB floor, or None to skip.

    Scalar → use as-is. Dict → pick entry for the current GPU family
    (like ``humaneval_score_threshold_amd`` for accuracy).
    """
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, dict):
        family = gpu_family if gpu_family is not None else detect_gpu_family()
        if family is None:
            logger.warning(
                "min_kv_buffer_mb is per-GPU %s but GPU family unknown; skip",
                list(spec.keys()),
            )
            return None
        if family not in spec:
            logger.warning(
                "min_kv_buffer_mb keys %s have no entry for %s; skip",
                list(spec.keys()),
                family,
            )
            return None
        return resolve_threshold(spec[family], gpu_family=family)
    logger.warning("min_kv_buffer_mb has unsupported type %s", type(spec).__name__)
    return None


def _owner_label(owner: FloorOwner) -> str:
    if isinstance(owner, type):
        return f"{owner.__module__}.{owner.__qualname__}"
    return getattr(owner, "__name__", repr(owner))


def find_active_test_owner() -> Optional[FloorOwner]:
    """Walk the stack for a TestCase class (or __main__) that set a threshold."""
    for frame_info in inspect.stack(context=0):
        cls = frame_info.frame.f_locals.get("cls")
        if not isinstance(cls, type):
            continue
        try:
            if not issubclass(cls, unittest.TestCase):
                continue
        except TypeError:
            continue
        if _raw_threshold(cls) is not None:
            return cls
    main = sys.modules.get("__main__")
    if main is not None and _raw_threshold(main) is not None:
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


def maybe_check_server_memory(
    base_url: str,
    *,
    api_key: Optional[str] = None,
    process: Any = None,
    owner: Optional[FloorOwner] = None,
) -> None:
    """Assert ``kv_buffer_mb >= min_kv_buffer_mb`` when a threshold is set."""
    if not memory_threshold_check_enabled():
        return

    pid = getattr(process, "pid", None) if process is not None else None
    if pid is not None:
        with _lock:
            if int(pid) in _CHECKED_PIDS:
                return

    owner = owner or find_active_test_owner()
    if owner is None:
        return
    threshold = resolve_threshold(_raw_threshold(owner))
    if threshold is None:
        return

    try:
        observed = fetch_kv_buffer_mb(base_url, api_key=api_key)
    except Exception as e:
        logger.warning("KV buffer check skipped: /server_info failed (%s)", e)
        return
    if observed is None:
        logger.warning("KV buffer check skipped: no kv_buffer_mb in server_info")
        return

    label = _owner_label(owner)
    logger.info(
        "KV buffer check %s: observed=%.1f threshold=%.1f",
        label,
        observed,
        threshold,
    )
    if observed < threshold:
        raise AssertionError(
            f"KV buffer capacity regression ({label}): "
            f"kv_buffer_mb={observed:g} < min_kv_buffer_mb={threshold:g}"
        )
    if pid is not None:
        with _lock:
            _CHECKED_PIDS.add(int(pid))


def reset_checked_pids() -> None:
    with _lock:
        _CHECKED_PIDS.clear()
