import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Runs on both H200 and B200 via nightly-8-gpu-common suite
# Higher est_time due to 6 variants with both performance and accuracy tests
register_cuda_ci(est_time=1800, suite="nightly-8-gpu-common", nightly=True)

GPT_OSS_120B_MXFP4_MODEL_PATH = "openai/gpt-oss-120b"
GPT_OSS_120B_EAGLE3_DRAFT_MODEL_PATH = "lmsys/EAGLE3-gpt-oss-120b-bf16"

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# multi-gpu floors; b200:samples=[1, 1], h200:samples=[1, 1] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = {
    # samples=[1, 1]
    "b200": [
        {
            "token_capacity": 17114423,
            "kv_cache_gb": 73.4382,
            "swa_size": 15594416,
            "full_size": 19493020,
            "swa_mem_gb": 150.579,
        },
        {
            "token_capacity": 17114423,
            "kv_cache_gb": 8.1576,
            "swa_size": 13691525,
            "full_size": 17114423,
            "swa_mem_gb": 132.2046,
        },
    ],
    # samples=[1, 1]
    "h200": [
        {
            "token_capacity": 13376204,
            "kv_cache_gb": 57.4002,
            "swa_size": 12177162,
            "full_size": 15221452,
            "swa_mem_gb": 117.5823,
        },
        {
            "token_capacity": 13376204,
            "kv_cache_gb": 6.3756,
            "swa_size": 10700963,
            "full_size": 13376204,
            "swa_mem_gb": 103.3263,
        },
    ],
}
# --- MEMORY_CAPACITY_FLOORS end ---


class TestGptOss120B(unittest.TestCase):
    """Unified test class for GPT-OSS-120B performance and accuracy.

    Testing:
    - Basic configs for MXFP4
    - Full config for MXFP4 with reasoning-parser, tool-call-parser, and MTP
    """

    def test_gpt_oss_120b_all_variants(self):
        """Run performance and accuracy for all GPT-OSS-120B variants."""
        base_args = [
            "--tp=8",
            "--trust-remote-code",
            "--cuda-graph-max-bs-decode=200",
            "--mem-fraction-static=0.93",
        ]
        # Lower batch size for EAGLE3 variants to avoid OOM
        base_args_eagle3 = [
            "--tp=8",
            "--trust-remote-code",
            "--cuda-graph-max-bs-decode=100",
            "--mem-fraction-static=0.85",
        ]
        parser_args = [
            "--reasoning-parser=gpt-oss",
            "--tool-call-parser=gpt-oss",
        ]
        eagle3_args = [
            "--speculative-algorithm=EAGLE3",
            f"--speculative-draft-model-path={GPT_OSS_120B_EAGLE3_DRAFT_MODEL_PATH}",
            "--speculative-num-steps=3",
            "--speculative-eagle-topk=1",
            "--speculative-num-draft-tokens=4",
        ]
        eagle3_env = {
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        }

        variants = [
            # Variant 1: MXFP4 baseline
            ModelLaunchSettings(
                GPT_OSS_120B_MXFP4_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
                variant="MXFP4",
            ),
            # Variant 2: MXFP4 + Parsers + EAGLE3 (full featured quantized, lower batch size)
            ModelLaunchSettings(
                GPT_OSS_120B_MXFP4_MODEL_PATH,
                tp_size=8,
                extra_args=base_args_eagle3 + parser_args + eagle3_args,
                env=eagle3_env,
                variant="MXFP4+Parsers+EAGLE3",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="GPT-OSS-120B",
            accuracy_params=None,
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_gpt_oss_120b",
            ),
        )


if __name__ == "__main__":
    unittest.main()
