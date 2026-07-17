"""Unit tests for the kv_buffer_mb memory floor helper."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.memory_threshold import (
    claim_min_kv_buffer_mb,
    gpu_family_from_text,
    kv_buffer_mb_from_server_info,
    mean_floor,
    reset_floor_counters,
    resolve_min_kv_buffer_mb,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKvBufferFloor(CustomTestCase):
    def test_kv_buffer_mb_from_server_info(self):
        info = {
            "internal_states": [
                {
                    "memory_usage": {
                        "weight": 10.0,  # must NOT be included
                        "kvcache": 2.0,
                        "graph": 3.0,  # must NOT be included
                        "mamba": 0.5,
                        "kv_buffer_mb": 2560.0,
                    }
                }
            ]
        }
        self.assertEqual(kv_buffer_mb_from_server_info(info), 2560.0)

    def test_kv_buffer_mb_fallback_excludes_weight_graph(self):
        info = {
            "memory_usage": {
                "weight": 10.0,
                "kvcache": 1.0,
                "graph": 5.0,
                "mamba": 1.0,
            }
        }
        # Only kvcache + mamba = 2 GB * 1024
        self.assertEqual(kv_buffer_mb_from_server_info(info), 2048.0)

    def test_gpu_family(self):
        self.assertEqual(gpu_family_from_text("NVIDIA H200"), "h200")
        self.assertEqual(gpu_family_from_text("8-gpu-b200"), "b200")

    def test_resolve_and_claim(self):
        self.assertEqual(resolve_min_kv_buffer_mb(100), [100.0])
        self.assertEqual(
            resolve_min_kv_buffer_mb({"h200": 10, "b200": 20}, gpu_family="b200"),
            [20.0],
        )

        class _T(CustomTestCase):
            min_kv_buffer_mb = [1000.0, 200.0]

        reset_floor_counters(_T)
        b = claim_min_kv_buffer_mb(_T, 210.0)
        a = claim_min_kv_buffer_mb(_T, 990.0)
        self.assertEqual(b, (200.0, 1))
        self.assertEqual(a, (1000.0, 0))

    def test_mean_floor(self):
        self.assertAlmostEqual(mean_floor([100.0, 100.0], factor=0.99), 99.0)


if __name__ == "__main__":
    unittest.main()
