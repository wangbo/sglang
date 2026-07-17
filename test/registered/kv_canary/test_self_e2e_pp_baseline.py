from __future__ import annotations

import unittest

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.pp_fixture import CanaryPPFixture

register_cuda_ci(est_time=220, stage="extra-a", runner_config="2-gpu-large")
register_amd_ci(est_time=243, stage="extra-a", runner_config="2-gpu-large-amd")

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# suite=extra-a-test-2-gpu-large samples=[3] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {
        "token_capacity": 81100,
        "kv_cache_gb": 1.089,
        "swa_size": 16220,
        "full_size": 81100,
        "swa_mem_gb": 1.5147,
    },
]
# --- MEMORY_CAPACITY_FLOORS end ---


class TestPPBaselineSwa(CanaryPPFixture):

    kv_canary_mode = CanaryMode.LOG

    def test_no_violation(self) -> None:
        for _ in range(self.workload_n_batches):
            self.send_parallel_requests()
        self.assert_no_violation(wait_seconds=2.0)
        self.maybe_assert_swa_divergence_observed()


if __name__ == "__main__":
    unittest.main()
