import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_server_kits import SpecLogprobKit
from sglang.test.server_fixtures.ngram_fixture import NgramServerBase

# Per-commit: Paged backend only.
# - FA3 base test archived to test/manual/spec/test_spec_ngram_fa3.py
# - Triton + Flashinfer moved to test_spec_ngram_extra.py
register_cuda_ci(est_time=400, stage="base-b", runner_config="1-gpu-large")

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# gpu=h100 samples=[3] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {"token_capacity": 898824, "kv_cache_gb": 48.015},
]
# --- MEMORY_CAPACITY_FLOORS end ---


class TestNgramSpeculativeDecodingPaged(NgramServerBase, GSM8KMixin, SpecLogprobKit):
    attention_backend = "flashinfer"
    extra_args = ["--page-size", "64"]


if __name__ == "__main__":
    unittest.main()
