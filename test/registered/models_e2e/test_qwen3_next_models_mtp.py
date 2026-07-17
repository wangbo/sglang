import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=290, stage="base-c", runner_config="4-gpu-h100")

QWEN3_NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# suite=base-c-test-4-gpu-h100 samples=[3, 3] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {
        "token_capacity": 1064013,
        "kv_cache_gb": 1.0098,
        "mamba_cache_size": 233,
        "mamba_conv_gb": 0.099,
        "mamba_ssm_gb": 4.1283,
    },
    {
        "token_capacity": 1188515,
        "kv_cache_gb": 1.1286,
        "mamba_cache_size": 336,
        "mamba_conv_gb": 0.1386,
        "mamba_ssm_gb": 5.9301,
    },
]
# --- MEMORY_CAPACITY_FLOORS end ---


class TestQwen3NextMTPTopk(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    # topk > 1 (tree) MTP on a hybrid-GDN model, on spec v2: the tree-aware mamba
    # state update lives in the spec v2 verify path, so mamba + topk > 1 no longer
    # falls back to spec v1.
    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.008
    other_args = [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "4",
        "--speculative-num-draft-tokens",
        "8",
        "--mem-fraction-static",
        "0.8",
        "--tp",
        "4",
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "128",
    ]


class TestQwen3NextMTPV2(GSM8KMixin, KLDivergenceMixin, DefaultServerBase):
    model = QWEN3_NEXT_MODEL
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.0035
    other_args = [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mem-fraction-static",
        "0.8",
        "--tp",
        "4",
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "128",
    ]


if __name__ == "__main__":
    unittest.main()
