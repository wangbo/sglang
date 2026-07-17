"""
Usage:
cd test/srt
python3 -m unittest test_qwen3_next_deterministic.TestFlashInferDeterministic
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_deterministic_utils import (
    COMMON_SERVER_ARGS,
    TestDeterministicBase,
)

register_cuda_ci(est_time=200, suite="nightly-4-gpu", nightly=True)

QWEN3_NEXT = "Qwen/Qwen3-Next-80B-A3B-Instruct"

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# gpu=h100 samples=[1] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {
        "token_capacity": 1216800,
        "kv_cache_gb": 13.9194,
        "mamba_cache_size": 695,
        "mamba_conv_gb": 0.2871,
        "mamba_ssm_gb": 12.2562,
    },
]
# --- MEMORY_CAPACITY_FLOORS end ---


class TestFlashInferDeterministic(TestDeterministicBase):
    @classmethod
    def get_model(cls):
        return QWEN3_NEXT

    # Test with flashinfer attention backend
    @classmethod
    def get_server_args(cls):
        args = COMMON_SERVER_ARGS
        args.extend(["--attention-backend", "flashinfer", "--tp", "4"])
        return args


class TestTritonDeterministic(TestDeterministicBase):
    @classmethod
    def get_model(cls):
        return QWEN3_NEXT

    # Test with triton attention backend
    @classmethod
    def get_server_args(cls):
        args = COMMON_SERVER_ARGS
        args.extend(["--attention-backend", "triton", "--tp", "4"])
        return args


if __name__ == "__main__":
    unittest.main()
