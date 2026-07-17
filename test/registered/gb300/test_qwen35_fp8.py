import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cuda_ci(est_time=7200, suite="nightly-4-gpu-gb300-qwen35-fp8", nightly=True)

MODEL_PATH = "Qwen/Qwen3.5-397B-A17B-FP8"

COMMON_ARGS = [
    "--trust-remote-code",
    "--reasoning-parser=qwen3",
    "--tool-call-parser=qwen3_coder",
    "--enable-flashinfer-allreduce-fusion",
    "--attention-backend=trtllm_mha",
    "--mem-fraction-static=0.8",
    "--mamba-scheduler-strategy=extra_buffer",
    "--enable-multimodal",
    "--enable-metrics",
]

TP_MTP_ARGS = [
    "--speculative-algorithm=EAGLE",
    "--speculative-num-steps=3",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=4",
]

DP_MTP_ARGS = [
    "--speculative-algorithm=EAGLE",
    "--speculative-num-steps=1",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=2",
]

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# gpu=gb300 samples=[1, 1] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {
        "token_capacity": 5699485,
        "kv_cache_gb": 5.445,
        "mamba_cache_size": 718,
        "mamba_conv_gb": 0.5544,
        "mamba_ssm_gb": 31.6305,
    },
    {
        "token_capacity": 2479783,
        "kv_cache_gb": 4.7322,
        "mamba_cache_size": 213,
        "mamba_conv_gb": 0.6633,
        "mamba_ssm_gb": 37.7586,
    },
]
# --- MEMORY_CAPACITY_FLOORS end ---


class TestQwen35Fp8(unittest.TestCase):
    """Qwen3.5-397B FP8 on GB300 (4x GB300 NVL4, tp=4)."""

    def test_qwen35_fp8(self):
        variants = [
            ModelLaunchSettings(
                MODEL_PATH,
                tp_size=4,
                extra_args=COMMON_ARGS + TP_MTP_ARGS,
                variant="TP4+MTP",
            ),
            ModelLaunchSettings(
                MODEL_PATH,
                tp_size=4,
                extra_args=COMMON_ARGS
                + ["--dp-size=4", "--enable-dp-attention"]
                + DP_MTP_ARGS,
                variant="TP4+DP4+DPA+MTP",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Qwen3.5-397B-FP8",
            accuracy_params=AccuracyTestParams(
                dataset="mmmu-pro", baseline_accuracy=0.76, repeat=1, max_tokens=32768
            ),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_gb300",
            ),
        )


if __name__ == "__main__":
    unittest.main()
