"""
Usage:
cd test/srt
python3 -m unittest test_qwen35_deterministic.TestQwen35Fa3Deterministic
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_deterministic_utils import (
    COMMON_SERVER_ARGS,
    TestDeterministicBase,
)

register_cuda_ci(est_time=360, stage="extra-b", runner_config="4-gpu-h100")

QWEN35 = "Qwen/Qwen3.5-35B-A3B"

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# gpu=h100 samples=[3] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {
        "token_capacity": 2476576,
        "kv_cache_gb": 23.6214,
        "mamba_cache_size": 1417,
        "mamba_conv_gb": 0.4851,
        "mamba_ssm_gb": 20.7801,
    },
]
# --- MEMORY_CAPACITY_FLOORS end ---


class TestQwen35Fa3Deterministic(TestDeterministicBase):
    @classmethod
    def get_model(cls):
        return QWEN35

    @classmethod
    def get_server_args(cls):
        return list(COMMON_SERVER_ARGS) + [
            "--tp",
            "4",
            "--attention-backend",
            "fa3",
            "--skip-server-warmup",
            "--mamba-scheduler-strategy",
            "extra_buffer",
            "--enable-flashinfer-allreduce-fusion",
            "--tokenizer-worker-num",
            "6",
            "--mem-fraction-static",
            "0.8",
        ]


if __name__ == "__main__":
    unittest.main()
