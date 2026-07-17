"""Unit tests for the total_mb memory floor helper."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.memory_threshold import (
    claim_min_total_mb,
    gpu_family_from_text,
    mean_floor,
    reset_floor_counters,
    resolve_min_total_memory_mb,
    total_mb_from_server_info,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTotalMbFloor(CustomTestCase):
    def test_total_mb_from_server_info(self):
        info = {
            "internal_states": [
                {
                    "memory_usage": {
                        "weight": 1.0,
                        "kvcache": 2.0,
                        "graph": 0.5,
                        "total_mb": 3584.0,
                    }
                }
            ]
        }
        self.assertEqual(total_mb_from_server_info(info), 3584.0)

    def test_total_mb_fallback_from_gb_components(self):
        info = {"memory_usage": {"weight": 1.0, "kvcache": 1.0, "graph": 0.0}}
        # 2 GB * 1024 = 2048 MB
        self.assertEqual(total_mb_from_server_info(info), 2048.0)

    def test_gpu_family(self):
        self.assertEqual(gpu_family_from_text("NVIDIA H200"), "h200")
        self.assertEqual(gpu_family_from_text("8-gpu-b200"), "b200")
        self.assertEqual(gpu_family_from_text("base-b-test-1-gpu-small"), "5090")

    def test_resolve_scalar_list_dict(self):
        self.assertEqual(resolve_min_total_memory_mb(100), [100.0])
        self.assertEqual(resolve_min_total_memory_mb([1, 2]), [1.0, 2.0])
        self.assertEqual(
            resolve_min_total_memory_mb({"h200": 10, "b200": 20}, gpu_family="b200"),
            [20.0],
        )
        self.assertIsNone(resolve_min_total_memory_mb({"h200": 10}, gpu_family="b200"))

    def test_claim_matches_closest(self):
        class _T(CustomTestCase):
            min_total_memory_mb = [1000.0, 200.0]

        reset_floor_counters(_T)
        b = claim_min_total_mb(_T, 210.0)
        a = claim_min_total_mb(_T, 990.0)
        self.assertEqual(b, (200.0, 1))
        self.assertEqual(a, (1000.0, 0))

    def test_mean_floor(self):
        self.assertAlmostEqual(mean_floor([100.0, 100.0], factor=0.99), 99.0)


if __name__ == "__main__":
    unittest.main()
