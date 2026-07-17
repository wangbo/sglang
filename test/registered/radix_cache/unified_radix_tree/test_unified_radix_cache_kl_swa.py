import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import UnifiedRadixTreeTestMixin
from sglang.test.kl_multiturn_utils import get_input_ids
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=250, stage="base-b", runner_config="2-gpu-large")

SWA_MODEL = "openai/gpt-oss-20b"

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# suite=base-b-test-2-gpu-large samples=[3] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {
        "token_capacity": 2201957,
        "kv_cache_gb": 25.2054,
        "swa_size": 1761565,
        "full_size": 2201957,
        "swa_mem_gb": 45.3618,
    },
]
# --- MEMORY_CAPACITY_FLOORS end ---


class TestUnifiedSWARadixCache(UnifiedRadixTreeTestMixin, CustomTestCase):
    """SWA hybrid + UnifiedRadixCache."""

    kl_threshold = 0.03
    gsm8k_threshold = 0.7
    mmlu_threshold = 0.7

    @unittest.skipIf(is_in_ci(), "SWA model mmlu eval not stable enough")
    def test_mmlu(self):
        super().test_mmlu()

    @classmethod
    def setUpClass(cls):
        cls.model = SWA_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "2",
                "--mem-fraction-static",
                "0.7",
                "--disable-piecewise-cuda-graph",
            ],
            env={"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
