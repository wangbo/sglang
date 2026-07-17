import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.mmmu_vlm_kit import MMMUMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.server_fixtures.mmmu_fixture import MMMUServerBase

register_cuda_ci(est_time=200, stage="extra-a", runner_config="2-gpu-large")

MODEL = "mistralai/Mistral-Small-4-119B-2603"

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# suite=extra-a-test-2-gpu-large samples=[3, 3] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {"token_capacity": 173383, "kv_cache_gb": 3.7224},
    {"token_capacity": 201578, "kv_cache_gb": 4.3263},
]
# --- MEMORY_CAPACITY_FLOORS end ---


class TestMistralSmall4TextOnly(GSM8KMixin, DefaultServerBase):
    gsm8k_accuracy_thres = 0.9
    model = MODEL
    other_args = ["--tp-size", "2", "--trust-remote-code"]


class TestMistralSmall4MMMU(MMMUMixin, MMMUServerBase):
    accuracy = 0.45
    model = MODEL
    other_args = ["--tp-size", "2", "--trust-remote-code"]
    mmmu_args = ["--limit=0.1"]
    """`--limit=0.1`: 10 percent of each task - this is fine for testing since the nominal result isn't interesting - this run is just to prevent relative regressions."""


if __name__ == "__main__":
    unittest.main()
