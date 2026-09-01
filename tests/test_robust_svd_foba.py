import unittest

from robust_svd_foba import RobustPoint, choose_robust_dose, robust_foba


def point(dose, a, b, protected=24):
    return RobustPoint(
        dose=dose,
        target_by_distribution={"a": a, "b": b},
        protected_by_distribution={
            "a": {"clean": protected, "quoted_attack": 24, "ambiguous": 24},
            "b": {"clean": 24, "quoted_attack": 24, "ambiguous": 24},
        },
    )


class RobustSvdFobaTest(unittest.TestCase):
    def test_dose_uses_worst_distribution_before_total(self):
        selected = choose_robust_dose({1.0: point(1.0, 20, 1), 2.0: point(2.0, 8, 8)})
        self.assertEqual(selected.dose, 2.0)

    def test_infeasible_high_repair_is_rejected(self):
        selected = choose_robust_dose({1.0: point(1.0, 20, 20, 21), 2.0: point(2.0, 7, 7)})
        self.assertEqual(selected.dose, 2.0)

    def test_forward_search_uses_robust_pair(self):
        values = {
            frozenset(): point(0.0, 0, 0),
            frozenset({"x"}): point(1.0, 8, 1),
            frozenset({"y"}): point(1.0, 4, 4),
            frozenset({"z"}): point(1.0, 2, 3),
            frozenset({"x", "y"}): point(1.0, 9, 7),
            frozenset({"x", "z"}): point(1.0, 8, 3),
            frozenset({"y", "z"}): point(1.0, 6, 6),
            frozenset({"x", "y", "z"}): point(1.0, 10, 8),
        }
        result = robust_foba(("x", "y", "z"), values.__getitem__, maximum_size=3)
        self.assertEqual(result["selected"], ["x", "y", "z"])
        self.assertEqual(result["point"]["target_by_distribution"], {"a": 10, "b": 8})


if __name__ == "__main__":
    unittest.main()
