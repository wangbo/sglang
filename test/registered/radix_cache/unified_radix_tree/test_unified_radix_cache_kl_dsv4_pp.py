import unittest

import test_unified_radix_cache_kl_dsv4 as dsv4_kl

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=900, stage="extra-b", runner_config="8-gpu-h200")

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# gpu=h200 samples=[3, 3, 3, 3, 3, 3, 2, 1] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {
        "token_capacity": 29133603,
        "dsv4_full": 29133603,
        "dsv4_swa": 7283358,
        "dsv4_c4": 7283400,
        "dsv4_c128": 227606,
        "dsv4_c4_state": 455209,
        "dsv4_c128_state": 0,
    },
    {
        "token_capacity": 31721057,
        "dsv4_full": 31721057,
        "dsv4_swa": 7930222,
        "dsv4_c4": 7930264,
        "dsv4_c128": 247820,
        "dsv4_c4_state": 495638,
        "dsv4_c128_state": 0,
    },
    {
        "token_capacity": 27435217,
        "dsv4_full": 27435217,
        "dsv4_swa": 6858762,
        "dsv4_c4": 6858804,
        "dsv4_c128": 214337,
        "dsv4_c4_state": 428672,
        "dsv4_c128_state": 0,
    },
    {
        "token_capacity": 28323947,
        "dsv4_full": 28323947,
        "dsv4_swa": 7080944,
        "dsv4_c4": 7080986,
        "dsv4_c128": 221280,
        "dsv4_c4_state": 442559,
        "dsv4_c128_state": 0,
    },
    {
        "token_capacity": 31720803,
        "dsv4_full": 31720803,
        "dsv4_swa": 7930137,
        "dsv4_c4": 7930200,
        "dsv4_c128": 247818,
        "dsv4_c4_state": 495633,
        "dsv4_c128_state": 0,
    },
    {
        "token_capacity": 21180741,
        "dsv4_full": 21180741,
        "dsv4_swa": 5295121,
        "dsv4_c4": 5295185,
        "dsv4_c128": 165474,
        "dsv4_c4_state": 330945,
        "dsv4_c128_state": 0,
    },
    {
        "token_capacity": 14617278,
        "dsv4_full": 14617278,
        "dsv4_swa": 3654224,
        "dsv4_c4": 3654319,
        "dsv4_c128": 114197,
        "dsv4_c4_state": 228389,
        "dsv4_c128_state": 0,
    },
    {
        "token_capacity": 19768,
        "dsv4_full": 19768,
        "dsv4_swa": 4815,
        "dsv4_c4": 4942,
        "dsv4_c128": 154,
        "dsv4_c4_state": 300,
        "dsv4_c128_state": 0,
    },
]
# --- MEMORY_CAPACITY_FLOORS end ---


class TestUnifiedDeepSeekV4FlashHiCachePP4TP2(
    dsv4_kl.TestUnifiedDeepSeekV4FlashHiCache
):
    """DeepSeek V4 Flash FP8 + HiCache + UnifiedRadixCache under PP4 TP2."""

    pp_size = 4
    tp_size = 2

    @unittest.skip("PP4TP2 coverage uses accuracy and cache-hit KL cases.")
    def test_multiturn_logprobs_match(self):
        pass


if __name__ == "__main__":
    unittest.main()
