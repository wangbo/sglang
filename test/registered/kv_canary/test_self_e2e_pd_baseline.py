from __future__ import annotations

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.pd_fixture import CanaryPDFixture

register_cuda_ci(est_time=180, stage="extra-a", runner_config="2-gpu-large")
register_amd_ci(est_time=106, stage="extra-a", runner_config="2-gpu-large-amd")

# --- MIN_KV_BUFFER_MB begin (auto; update_memory_thresholds.py) ---
# gpu=h100 updated=2026-07-18
MIN_KV_BUFFER_MB = 68307.1
# --- MIN_KV_BUFFER_MB end ---


class TestPDBaselineMha(CanaryPDFixture):
    model_mode = "mha"

    def test_clean_pd_run_produces_no_canary_violation_on_either_side(self) -> None:
        self.send_parallel_short_requests(n=4)
        self.assert_no_violation(side="prefill")
        self.assert_no_violation(side="decode")


class TestPDBaselineSwa(CanaryPDFixture):
    model_mode = "swa"

    def test_clean_pd_run_produces_no_canary_violation_on_either_side(self) -> None:
        self.send_parallel_short_requests(n=4)
        self.assert_no_violation(side="prefill")
        self.assert_no_violation(side="decode")


if __name__ == "__main__":
    unittest.main()
