import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modal_causal_selector_cascade_diagnostic.py"
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


CALIBRATION_NODE = next(
    node for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "calibrated_candidate"
)
NAMESPACE = {"METHOD_PRIORITY": constant("METHOD_PRIORITY")}
exec(compile(ast.Module(body=[CALIBRATION_NODE], type_ignores=[]), str(PATH), "exec"), NAMESPACE)
calibrated_candidate = NAMESPACE["calibrated_candidate"]


def result_with_passes(passes):
    budgets = (64, 128, 192)
    methods = constant("METHOD_PRIORITY")
    return {
        "budgets": budgets,
        "curve": {
            str(budget): {
                method: {
                    "behavioral_pass": (budget, method) in passes,
                    "support": [f"{method}:{budget}"],
                }
                for method in methods
            }
            for budget in budgets
        },
    }


class CausalSelectorCascadeDiagnosticTest(unittest.TestCase):
    def test_protocol_hash_is_frozen(self):
        expected = constant("PROTOCOL_SHA256")
        actual = hashlib.sha256(
            (ROOT / "CAUSAL_SELECTOR_CASCADE_DIAGNOSTIC.md").read_bytes()
        ).hexdigest()
        self.assertEqual(actual, expected)

    def test_smallest_stable_budget_wins_before_method_priority(self):
        result = result_with_passes({
            (64, "foba64_svd"), (128, "foba64_svd"),
            (128, "top_svd"), (192, "top_svd"),
        })
        candidate = calibrated_candidate(result)
        self.assertEqual(candidate["budget"], 64)
        self.assertEqual(candidate["method"], "foba64_svd")

    def test_method_priority_breaks_same_budget_tie(self):
        result = result_with_passes({
            (64, "top_svd"), (128, "top_svd"),
            (64, "foba64_svd"), (128, "foba64_svd"),
        })
        self.assertEqual(calibrated_candidate(result)["method"], "top_svd")

    def test_unstable_candidate_abstains(self):
        result = result_with_passes({(64, "top_svd"), (192, "top_svd")})
        self.assertIsNone(calibrated_candidate(result))


if __name__ == "__main__":
    unittest.main()
