"""EAGLE3 spec-decoding core: overlap (spec v2) x no-overlap (spec v1) matrix,
same standard config (topk=1, page_size=1), only ``disable_overlap`` differs.
flashinfer is pinned (the 5090 default) so a default-selection change can't
silently alter what this exercises.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecCorrectnessKit,
    SpecFeatureKit,
    SpecLogprobKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base

register_cuda_ci(est_time=480, stage="base-b", runner_config="1-gpu-small")

_KITS = (
    SpecCorrectnessKit,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
    MatchedStopMixin,
)

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# suite=base-b-test-1-gpu-small samples=[4, 1] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {"token_capacity": 72657, "kv_cache_gb": 8.8704},
    {"token_capacity": 72658, "kv_cache_gb": 8.8704},
]
# --- MEMORY_CAPACITY_FLOORS end ---


class _Core(Eagle3Base):
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagle3Overlap(_Core, *_KITS):
    """Spec v2 (overlap scheduler on)."""

    disable_overlap = False


class TestEagle3NoOverlap(_Core, *_KITS):
    """Spec v1 (overlap scheduler off)."""

    disable_overlap = True


if __name__ == "__main__":
    unittest.main()
