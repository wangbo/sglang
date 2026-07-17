import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

# Runs on both H200 and B200 via nightly-8-gpu-common suite
register_cuda_ci(est_time=1800, suite="nightly-8-gpu-common", nightly=True)

QWEN35_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B-FP8"

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# multi-gpu floors; b200:samples=[1, 1, 1], h200:samples=[1, 1, 1] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = {
    # samples=[1, 1, 1]
    "b200": [
        {
            "token_capacity": 3418605,
            "kv_cache_gb": 48.906,
            "mamba_cache_size": 1967,
            "mamba_conv_gb": 0.7623,
            "mamba_ssm_gb": 43.2432,
        },
        {
            "token_capacity": 1565049,
            "kv_cache_gb": 44.7678,
            "mamba_cache_size": 224,
            "mamba_conv_gb": 0.693,
            "mamba_ssm_gb": 39.6792,
        },
        {
            "token_capacity": 1952121,
            "kv_cache_gb": 3.7224,
            "mamba_cache_size": 119,
            "mamba_conv_gb": 0.3762,
            "mamba_ssm_gb": 21.2355,
        },
    ],
    # samples=[1, 1, 1]
    "h200": [
        {
            "token_capacity": 2305676,
            "kv_cache_gb": 32.9868,
            "mamba_cache_size": 1326,
            "mamba_conv_gb": 0.5148,
            "mamba_ssm_gb": 29.1753,
        },
        {
            "token_capacity": 1012264,
            "kv_cache_gb": 28.9674,
            "mamba_cache_size": 144,
            "mamba_conv_gb": 0.4455,
            "mamba_ssm_gb": 25.5816,
        },
        {
            "token_capacity": 1190941,
            "kv_cache_gb": 2.277,
            "mamba_cache_size": 75,
            "mamba_conv_gb": 0.2376,
            "mamba_ssm_gb": 13.4046,
        },
    ],
}
# --- MEMORY_CAPACITY_FLOORS end ---


class TestQwen35(unittest.TestCase):
    """Unified test class for Qwen3.5-397B-A17B performance and accuracy.

    Qwen3.5 is a 397B MoE VLM with 17B active params.
    Features hybrid reasoning, tool calling, and multimodal capabilities.
    Runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with gsm8k)
    """

    def test_qwen35(self):
        """Run performance and accuracy for Qwen3.5-397B-A17B."""
        base_args = [
            "--trust-remote-code",
            "--reasoning-parser=qwen3",
            "--tool-call-parser=qwen3_coder",
            "--mem-fraction-static=0.8",
        ]
        dp_args = ["--dp=8", "--enable-dp-attention"]
        mtp_args = [
            "--speculative-algorithm=EAGLE",
            "--speculative-num-steps=3",
            "--speculative-eagle-topk=1",
            "--speculative-num-draft-tokens=4",
            "--mamba-scheduler-strategy=extra_buffer",
        ]

        variants = [
            ModelLaunchSettings(
                QWEN35_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
                variant="TP8",
            ),
            ModelLaunchSettings(
                QWEN35_MODEL_PATH,
                tp_size=8,
                extra_args=base_args + dp_args,
                variant="TP8+DP8",
            ),
            ModelLaunchSettings(
                QWEN35_MODEL_PATH,
                tp_size=8,
                extra_args=base_args + dp_args + mtp_args,
                variant="TP8+DP8+MTP",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Qwen3.5-397B-A17B",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.95,
                thinking_mode="qwen3",
                max_tokens=8192,
                num_examples=200,
            ),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_qwen35",
            ),
        )


if __name__ == "__main__":
    unittest.main()
