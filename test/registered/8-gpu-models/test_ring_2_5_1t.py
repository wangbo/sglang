import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cuda_ci(est_time=510, suite="nightly-8-gpu-common", nightly=True)

RING_2_5_1T_MODEL_PATH = "inclusionAI/Ring-2.5-1T"

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# suite=nightly-8-gpu-common samples=[2] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {
        "token_capacity": 819159,
        "kv_cache_gb": 8.7912,
        "mamba_cache_size": 231,
        "mamba_conv_gb": 0.0,
        "mamba_ssm_gb": 7.9349,
    },
]
# --- MEMORY_CAPACITY_FLOORS end ---


class TestRing2_5_1T(unittest.TestCase):
    """Accuracy test for Ring-2.5-1T.

    Ring-2.5-1T is a ~1T MoE model with linear attention layers.
    Uses TP=8 for GSM8K evaluation.
    """

    def test_ring_2_5_1t(self):
        base_args = [
            "--trust-remote-code",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
            "--watchdog-timeout",
            "1800",
            "--soft-watchdog-timeout",
            "1800",
        ]

        variants = [
            ModelLaunchSettings(
                RING_2_5_1T_MODEL_PATH,
                tp_size=8,
                extra_args=base_args,
                variant="TP8",
                launch_timeout=1800,
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Ring-2.5-1T",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                num_examples=200,
                baseline_accuracy=0.88,
                temperature=1.2,
                top_p=0.8,
                max_tokens=4096,
            ),
        )


if __name__ == "__main__":
    unittest.main()
