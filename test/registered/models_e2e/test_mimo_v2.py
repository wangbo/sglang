import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.mmmu_fixture import MMMUServerBase

register_cuda_ci(est_time=400, stage="base-c", runner_config="8-gpu-h200")

MIMO_V2_MODEL = "XiaomiMiMo/MiMo-V2.5"
MIMO_V2_OTHER_ARGS = [
    "--tp",
    "8",
    "--dp",
    "2",
    "--enable-dp-attention",
    "--mm-enable-dp-encoder",
    "--attention-backend",
    "fa3",
    "--mm-attention-backend",
    "fa3",
    "--reasoning-parser",
    "mimo",
    "--enable-hierarchical-cache",
    "--hicache-ratio",
    "1.5",
    "--hicache-mem-layout",
    "page_first_direct",
    "--hicache-io-backend",
    "direct",
]
MIMO_V2_MTP_OTHER_ARGS = MIMO_V2_OTHER_ARGS + [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--enable-multi-layer-eagle",
]

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# gpu=h200 samples=[3, 3, 1, 1] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {
        "token_capacity": 1079933,
        "kv_cache_gb": 40.1676,
        "swa_size": 863946,
        "full_size": 1079933,
        "swa_mem_gb": 45.9591,
    },
    {
        "token_capacity": 1079933,
        "kv_cache_gb": 0.0,
        "swa_size": 863946,
        "full_size": 1079933,
        "swa_mem_gb": 1.0296,
    },
    {
        "token_capacity": 863903,
        "kv_cache_gb": 1.0296,
        "swa_size": 863903,
        "full_size": 1079880,
        "swa_mem_gb": 45.9558,
    },
    {
        "token_capacity": 1079880,
        "kv_cache_gb": 0.0,
        "swa_size": 863903,
        "full_size": 1079880,
        "swa_mem_gb": 1.0296,
    },
]
# --- MEMORY_CAPACITY_FLOORS end ---


class TestMiMoV2(GSM8KMixin, MMMUServerBase):
    gsm8k_accuracy_thres = 0.75
    gsm8k_accept_length_thres = 2.5
    model = MIMO_V2_MODEL
    mem_fraction_static = 0.65
    server_api_key = None
    other_args = MIMO_V2_MTP_OTHER_ARGS

    @classmethod
    def setUpClass(cls):
        with envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.override(True):
            super().setUpClass()


if __name__ == "__main__":
    unittest.main()
