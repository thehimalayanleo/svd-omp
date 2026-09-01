import unittest

from robust_svd_bridge_foba import bridge_foba, violation
from robust_svd_foba import RobustPoint


def point(a, b, protected=24, dose=1.0):
    return RobustPoint(
        dose=dose,
        target_by_distribution={"a": a, "b": b},
        protected_by_distribution={
            "a": {"clean": protected, "quoted": 24, "ambiguous": 24},
            "b": {"clean": 24, "quoted": 24, "ambiguous": 24},
        },
    )


class RobustSvdBridgeFobaTest(unittest.TestCase):
    def test_violation(self):
        self.assertEqual(violation(point(0, 0, 20)), 2)

    def test_bridge_crosses_infeasible_singleton(self):
        values = {
            frozenset(): point(0, 0, 21, 0.0),
            frozenset({"x"}): point(12, 0, 21),
            frozenset({"y"}): point(0, 0, 22),
            frozenset({"z"}): point(1, 0, 21),
            frozenset({"x", "y"}): point(12, 9, 22),
            frozenset({"x", "z"}): point(13, 0, 21),
            frozenset({"y", "z"}): point(2, 2, 22),
            frozenset({"x", "y", "z"}): point(12, 10, 22),
        }
        result = bridge_foba(("x", "y", "z"), values.__getitem__, maximum_size=3)
        self.assertEqual(result["selected"], ["x", "y", "z"])
        self.assertEqual(result["point"]["target_by_distribution"], {"a": 12, "b": 10})

    def test_requires_final_feasibility(self):
        values = {
            frozenset(): point(0, 0, 21, 0.0),
            frozenset({"x"}): point(5, 0, 21),
            frozenset({"y"}): point(4, 0, 21),
            frozenset({"x", "y"}): point(7, 1, 21),
        }
        with self.assertRaises(ValueError):
            bridge_foba(("x", "y"), values.__getitem__, maximum_size=2)


if __name__ == "__main__":
    unittest.main()
