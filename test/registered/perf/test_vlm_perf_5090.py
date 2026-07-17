"""
VLM Performance tests that work on 5090 (32GB) - VLM offline throughput and online latency tests.
"""

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    run_bench_serving,
    write_github_step_summary,
)

register_cuda_ci(est_time=406, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=500, suite="stage-b-test-1-gpu-small-amd")

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# suite=base-b-test-1-gpu-small samples=[3, 1] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {"token_capacity": 410372, "kv_cache_gb": 14.0976},
    {"token_capacity": 410373, "kv_cache_gb": 14.0976},
]
# --- MEMORY_CAPACITY_FLOORS end ---


class TestVLMPerf5090(CustomTestCase):
    def test_vlm_offline_throughput(self):
        res = run_bench_serving(
            model=DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
            num_prompts=200,
            request_rate=float("inf"),
            other_server_args=[
                "--mem-fraction-static",
                "0.7",
            ],
            dataset_name="mmmu",
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_vlm_offline_throughput (5090)\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            self.assertGreater(res["output_throughput"], 2000)

    def test_vlm_online_latency(self):
        res = run_bench_serving(
            model=DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
            num_prompts=250,
            request_rate=1,
            other_server_args=[
                "--mem-fraction-static",
                "0.7",
            ],
            dataset_name="mmmu",
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_vlm_online_latency (5090)\n"
                f"median_e2e_latency_ms: {res['median_e2e_latency_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_e2e_latency_ms"], 16500)
            self.assertLess(res["median_ttft_ms"], 150)
            self.assertLess(res["median_itl_ms"], 8)


if __name__ == "__main__":
    unittest.main()
