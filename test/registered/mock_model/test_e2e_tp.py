from __future__ import annotations

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.mock_model.utils import run_mock_model_bench_serving
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=600, stage="extra-a", runner_config="2-gpu-large")
register_amd_ci(est_time=167, stage="extra-a", runner_config="2-gpu-large-amd")

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# suite=extra-a-test-2-gpu-large samples=[3] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {"token_capacity": 1255060, "kv_cache_gb": 67.023},
]
# --- MEMORY_CAPACITY_FLOORS end ---


class TestE2ETensorParallel(CustomTestCase):
    def test_tp_no_canary_violation(self) -> None:
        run_mock_model_bench_serving(
            extra_server_args=["--tp", "2", "--mem-fraction-static", "0.88"],
        )


if __name__ == "__main__":
    unittest.main()
