import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.server_fixtures.dsa_mtp_fixture import (
    DsaMtpEvalConfigDefaults,
    DsaMtpServerBase,
)

register_cuda_ci(
    est_time=400,
    stage="base-c",
    runner_config="8-gpu-h200",
)

# --- MEMORY_CAPACITY_FLOORS begin (auto; update_memory_thresholds.py) ---
# gpu=h200 samples=[3] updated=2026-07-17
MEMORY_CAPACITY_FLOORS = [
    {"token_capacity": 199330, "kv_cache_gb": 18.6021},
]
# --- MEMORY_CAPACITY_FLOORS end ---


class TestGLM52TPMTP(
    DsaMtpServerBase, DsaMtpEvalConfigDefaults, GSM8KMixin, SpecDecodingMixin
):
    model = "zai-org/GLM-5.2-FP8"
    mem_fraction_static = 0.8
    bs_1_speed_thres = 150


if __name__ == "__main__":
    unittest.main()
