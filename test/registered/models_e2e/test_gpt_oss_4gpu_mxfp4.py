import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.gpt_oss_common import BaseTestGptOss

register_cuda_ci(est_time=220, stage="base-c", runner_config="4-gpu-h100")
register_cuda_ci(est_time=220, stage="base-c", runner_config="4-gpu-b200")

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# suite=base-c-test-4-gpu-h100 samples=[2] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {
        "token_capacity": 2870133,
        "kv_cache_gb": 24.6312,
        "swa_size": 2296107,
        "full_size": 2870133,
        "swa_mem_gb": 44.3421,
    },
]
# --- MEMORY_CAPACITY_FLOORS end ---


class TestGptOss4GpuMxfp4(BaseTestGptOss):
    def test_mxfp4_120b(self):
        self.run_test(
            model_variant="120b",
            quantization="mxfp4",
            expected_score_of_reasoning_effort={
                "low": 0.58,
            },
            other_args=[
                "--tp",
                "4",
                "--cuda-graph-max-bs-decode",
                "200",
            ],
        )


if __name__ == "__main__":
    unittest.main()
