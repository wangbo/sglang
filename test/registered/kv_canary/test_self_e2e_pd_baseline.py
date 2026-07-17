from __future__ import annotations

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.pd_fixture import CanaryPDFixture

register_cuda_ci(est_time=180, stage="extra-a", runner_config="2-gpu-large")
register_amd_ci(est_time=106, stage="extra-a", runner_config="2-gpu-large-amd")

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# suite=extra-a-test-2-gpu-large samples=[3, 3, 3, 3] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {"token_capacity": 608578, "kv_cache_gb": 65.0034},
    {"token_capacity": 616589, "kv_cache_gb": 65.8548},
    {
        "token_capacity": 1616285,
        "kv_cache_gb": 35.0526,
        "swa_size": 1312697,
        "full_size": 1640872,
        "swa_mem_gb": 56.9613,
    },
    {
        "token_capacity": 1616285,
        "swa_size": 1293027,
        "full_size": 1616285,
        "swa_mem_gb": 56.1066,
    },
]
# --- MEMORY_CAPACITY_FLOORS end ---


class TestPDBaselineMha(CanaryPDFixture):
    model_mode = "mha"

    def test_clean_pd_run_produces_no_canary_violation_on_either_side(self) -> None:
        self.send_parallel_short_requests(n=4)
        self.assert_no_violation(side="prefill")
        self.assert_no_violation(side="decode")


class TestPDBaselineSwa(CanaryPDFixture):
    model_mode = "swa"

    def test_clean_pd_run_produces_no_canary_violation_on_either_side(self) -> None:
        self.send_parallel_short_requests(n=4)
        self.assert_no_violation(side="prefill")
        self.assert_no_violation(side="decode")


if __name__ == "__main__":
    unittest.main()
