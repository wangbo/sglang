"""B200 per-commit CI: DeepSeek-V4-Flash FP4 (LowLatency recipe).

Launches TP=4 with flashinfer_mxfp4 MoE runner + EAGLE speculative decoding.
Runs 12 ServerSanity probes (correctness, streaming, concurrency, determinism)
plus a GSM8K accuracy gate.

Registry: extra-b-test-deepep-4-gpu-b200 (label-gated, 4x B200)
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=900, stage="extra-b", runner_config="deepep-4-gpu-b200")

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SERVER_LAUNCH_TIMEOUT = 3600


_W4A8_MEGAMOE_ENV = {
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "4096",
}


_W4A4_MEGAMOE_ENV = {
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "4096",
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS": "1",
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND": "1",
}

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# gpu=b200 samples=[3, 1] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {
        "token_capacity": 10715358,
        "dsv4_full": 10715358,
        "dsv4_swa": 1071375,
        "dsv4_c4": 2678839,
        "dsv4_c128": 83713,
        "dsv4_c4_state": 133921,
        "dsv4_c128_state": 0,
    },
    {
        "token_capacity": 10714176,
        "dsv4_full": 10714176,
        "dsv4_swa": 1071290,
        "dsv4_c4": 2678544,
        "dsv4_c128": 83704,
        "dsv4_c4_state": 133911,
        "dsv4_c128_state": 0,
    },
]
# --- MEMORY_CAPACITY_FLOORS end ---


class TestDSV4FlashFP4B200W4A8MegaMoE(
    SpecDecodingMixin,
    BasicDecodeCorrectnessMixin,
    GSM8KMixin,
    CustomTestCase,
):
    """Balanced recipe: TP=4, DP=4, MegaMoE."""

    gsm8k_accuracy_thres = 0.93
    accept_length_thres = 1.8
    bs_1_speed_thres = 100

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--dp",
                "4",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "megamoe",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
            ],
            env=_W4A8_MEGAMOE_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


class TestDSV4FlashFP4B200W4A4MegaMoE(
    SpecDecodingMixin,
    BasicDecodeCorrectnessMixin,
    GSM8KMixin,
    CustomTestCase,
):
    """Balanced recipe: TP=4, DP=4, MegaMoE."""

    gsm8k_accuracy_thres = 0.93
    accept_length_thres = 2.8
    bs_1_speed_thres = 100

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--dp",
                "4",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "megamoe",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
            ],
            env=_W4A4_MEGAMOE_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
