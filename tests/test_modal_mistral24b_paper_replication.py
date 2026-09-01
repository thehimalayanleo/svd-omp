import ast
import hashlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modal_mistral24b_paper_replication.py"
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


CONSENSUS_NODE = next(
    node for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "build_consensus"
)
CONSENSUS_NAMESPACE = {"SUPPORT_BUDGET": 224}
exec(compile(ast.Module(body=[CONSENSUS_NODE], type_ignores=[]), str(PATH), "exec"), CONSENSUS_NAMESPACE)
build_consensus = CONSENSUS_NAMESPACE["build_consensus"]


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class PaperReplicationRunnerTest(unittest.TestCase):
    def test_frozen_campaign_constants(self):
        self.assertEqual(constant("TRAINING_SEEDS"), (607, 613, 619))
        self.assertEqual(constant("SUPPORT_BUDGET"), 224)
        self.assertEqual(constant("RANDOM_SUPPORTS"), 999)
        self.assertEqual(constant("OMP_PREFIX"), 64)

    def test_protocol_and_data_hashes(self):
        paths = {
            constant("DEVELOPMENT").rsplit("/root/svd-omp/", 1)[1]:
                "cd8f982386a6a18460b4836d244d9cf4456bb4390ae51bc501612d161c8f18a5",
            constant("CONFIRMATION").rsplit("/root/svd-omp/", 1)[1]:
                "b186ba54aa06b78c5f79355fe94d5ff04fdfa35807b550e22a6b6041bfb60035",
            constant("PROTOCOL").rsplit("/root/svd-omp/", 1)[1]:
                "02d9b1a932632499f19217ef30cc911e8e07b6c0f0312da4a29f226d59ba053d",
        }
        for relative, expected in paths.items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)

    def test_consensus_prefers_frequency_before_strength(self):
        names = [f"atom:{index}" for index in range(300)]
        results = []
        for shift in (0, 1, 2):
            support = names[shift:shift + 224]
            results.append({
                "selection": {
                    "methods": {"foba64_svd160": support},
                    "singular_values": {
                        name: float(300 - index) for index, name in enumerate(names)
                    },
                }
            })
        consensus = build_consensus(results)
        self.assertEqual(len(consensus), 224)
        self.assertEqual(len(set(consensus)), 224)
        self.assertIn("atom:2", consensus)

    def test_development_image_does_not_mount_confirmation(self):
        development_block = SOURCE.split("development_image =", 1)[1].split(
            "confirmation_image =", 1
        )[0]
        self.assertNotIn("CONFIRMATION", development_block)


if __name__ == "__main__":
    unittest.main()
