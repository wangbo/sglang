from __future__ import annotations

import unittest

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.consts import SWA_POOL_SERVER_ARGS
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=236, stage="extra-a", runner_config="1-gpu-small-amd")

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# suite=extra-a-test-1-gpu-small samples=[3, 3] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {"token_capacity": 64880, "kv_cache_gb": 6.93},
    {
        "token_capacity": 81100,
        "kv_cache_gb": 1.089,
        "swa_size": 16220,
        "full_size": 81100,
        "swa_mem_gb": 1.5147,
    },
]
# --- MEMORY_CAPACITY_FLOORS end ---


class _BaselineBase(CanaryE2EBase):
    """No perturb, kv-canary=log, sweep off. Server should run clean with no canary
    violations and every request must come back 200."""

    kv_canary_mode = CanaryMode.LOG
    extra_env = {}

    @classmethod
    def setUpClass(cls) -> None:
        if cls is _BaselineBase:
            raise unittest.SkipTest("abstract base; concrete subclasses set model_mode")
        super().setUpClass()

    def test_no_violation(self) -> None:
        """Verify the baseline canary run completes without violations."""
        for _ in range(self.workload_n_batches):
            self.send_parallel_requests()
        self.assert_no_violation(wait_seconds=2.0)
        self.maybe_assert_swa_divergence_observed()


class TestBaselineMha(_BaselineBase):
    model_mode = "mha"


class TestBaselineSwa(_BaselineBase):
    model_mode = "swa"
    extra_server_args = SWA_POOL_SERVER_ARGS


if __name__ == "__main__":
    unittest.main()
