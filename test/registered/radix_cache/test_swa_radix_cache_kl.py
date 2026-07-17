import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

MODEL = "openai/gpt-oss-20b"

register_cuda_ci(est_time=151, stage="base-b", runner_config="1-gpu-large")

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# gpu=h100 samples=[3] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {
        "token_capacity": 959038,
        "kv_cache_gb": 21.9582,
        "swa_size": 767230,
        "full_size": 959038,
        "swa_mem_gb": 39.5109,
    },
]
# --- MEMORY_CAPACITY_FLOORS end ---


class TestSWARadixCacheKL(KLDivergenceMixin, DefaultServerBase):
    model = MODEL
    kl_div_thres = 0.02  # it was 0.002
    kl_div_decode_max_new_tokens = 2048
    other_args = [
        "--tp-size",
        "1",
        "--mem-fraction-static",
        "0.70",
        "--cuda-graph-backend-prefill=disabled",
    ]


if __name__ == "__main__":
    unittest.main()
