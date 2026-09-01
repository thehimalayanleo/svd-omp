#!/usr/bin/env python3
"""Recompute the paper-grade causal claims from retained campaign artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/behavioral_causal_audit"

CAMPAIGNS = {
    "fresh_mistral24b": {
        "stem": "mistral24b_paper_replication",
        "protocol": "MISTRAL24B_PAPER_REPLICATION_PROTOCOL.md",
        "development": "data/behavior_audit/mistral24b_paper_replication_development.jsonl",
        "confirmation": "data/behavior_audit/mistral24b_paper_replication_confirmation.jsonl",
        "seeds": (607, 613, 619),
        "model_revision": "68faf511d618ef198fef186659617cfd2eb8e33a",
        "parameters": 24_011_361_280,
        "atoms": 640,
        "budget": 224,
        "primary": "foba64_svd160",
        "confirmation_rows": 128,
        "evidence_class": "prospective_fixed_budget_replication",
    },
    "qwen30b": {
        "stem": "qwen30b_causal",
        "protocol": "QWEN30B_POSITION_BIAS_CAUSAL_PROTOCOL.md",
        "development": "data/behavior_audit/qwen30b_position_bias_development.jsonl",
        "confirmation": "data/behavior_audit/qwen30b_position_bias_confirmation.jsonl",
        "seeds": (811, 821, 823),
        "model_revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        "parameters": 30_532_122_624,
        "atoms": 768,
        "budget": 272,
        "primary": "foba64_svd208",
        "confirmation_rows": 128,
        "evidence_class": "prospective_cross_family_confirmation",
    },
    "metadata_abstention": {
        "stem": "mistral24b_metadata_abstention_v3_causal",
        "protocol": "MISTRAL24B_METADATA_ABSTENTION_V3_PROTOCOL.md",
        "development": "data/behavior_audit/mistral24b_metadata_abstention_v3_development.jsonl",
        "confirmation": "data/behavior_audit/mistral24b_metadata_abstention_v3_confirmation.jsonl",
        "seeds": (701, 709, 719),
        "model_revision": "68faf511d618ef198fef186659617cfd2eb8e33a",
        "parameters": 24_011_361_280,
        "atoms": 640,
        "budget": 224,
        "primary": "foba64_svd160",
        "confirmation_rows": 96,
        "evidence_class": "exploratory_post_screen_redesign",
    },
}

# Filled only after the Modal outputs were copied back. These seals cover every
# development summary, confirmation summary, and per-seed confirmation file.
ARTIFACT_HASHES = {
    "results/behavioral_causal_audit/mistral24b_paper_replication_development_summary.json": "ef4a18264199200611654608eefbd8fec8353ba43e3b0597f25feeb109ff034a",
    "results/behavioral_causal_audit/mistral24b_paper_replication_confirmation_summary.json": "f836587db30d183a9d916708e42b19da43b880f3194b335d1bf5aaaec4c44b02",
    "results/behavioral_causal_audit/mistral24b_paper_replication_confirmation_seed607.json": "8d1f4e2b8744c5796c553e996c7f237713840b50ac7d11c3560c98b20fc1c6fe",
    "results/behavioral_causal_audit/mistral24b_paper_replication_confirmation_seed613.json": "f521044fdadbdd256cc2f45fef693dfd41e5e622d0429602b6a16c913e3eff1e",
    "results/behavioral_causal_audit/mistral24b_paper_replication_confirmation_seed619.json": "7106bc6f28434dc9393519251cb7bf30ec58df06c0168a23efdb6418b06084b8",
    "results/behavioral_causal_audit/qwen30b_causal_development_summary.json": "9191dfa70bf12b294135ddc56c76baf87f62c34febc1cfa1468833e2d2804bb8",
    "results/behavioral_causal_audit/qwen30b_causal_confirmation_summary.json": "a577a0bf6aca78a84e2984df3484aa57bcf4b390359fc5307680873463a6dab4",
    "results/behavioral_causal_audit/qwen30b_causal_confirmation_seed811.json": "0b8f6c2744809c6656ba2c648db8d7522ec6ebd3b53c9c1eb93682bc1f8e5d3f",
    "results/behavioral_causal_audit/qwen30b_causal_confirmation_seed821.json": "57a52f7b811498009834b94306359f878cf96b577383294fe420e65177d230ad",
    "results/behavioral_causal_audit/qwen30b_causal_confirmation_seed823.json": "229b361f89cf233ff75fd5498871e5ca2e52eda29679503b8d6cfd89c08b215b",
    "results/behavioral_causal_audit/mistral24b_metadata_abstention_v3_causal_development_summary.json": "9d6c9b405daddabd5b2a70ad58f4ee650344c182240bc71aceb0e9417b66ceb2",
    "results/behavioral_causal_audit/mistral24b_metadata_abstention_v3_causal_confirmation_summary.json": "c3ca777b728abbb1b99cd8815b7c565ecefffb9d060c8e4d9ec31024a46e4ba3",
    "results/behavioral_causal_audit/mistral24b_metadata_abstention_v3_causal_confirmation_seed701.json": "a5e67767e7f5fc609c3e5c0559c4808c809fad2c099323528e934c30e57034b3",
    "results/behavioral_causal_audit/mistral24b_metadata_abstention_v3_causal_confirmation_seed709.json": "622443b87d1d5570440d84e1552d144d7be6ac1f9a46717fdced03cb7b6c8497",
    "results/behavioral_causal_audit/mistral24b_metadata_abstention_v3_causal_confirmation_seed719.json": "e432f2872346092857555ece6380efe3bb9554f00576e48c1128172f12560c83",
}

DIAGNOSTIC_HASHES = {
    "results/behavioral_causal_audit/qwen30b_dense_cycle_numeric_diagnostic_summary.json": "ee522106d3eafd87f213d4404685ed0dbc5cf0c4185e701cb02556d2a4d39c75",
    "results/behavioral_causal_audit/qwen30b_dense_cycle_numeric_diagnostic_seed811.json": "ba160b4b942b70fbfb267aabf051c0aa4436e1980716f6324c10d7dfdf2a4ef8",
    "results/behavioral_causal_audit/qwen30b_dense_cycle_numeric_diagnostic_seed821.json": "b2ff341193659ff0ddfeae4a79267bdcee671312d9bc94f46935a72c20886524",
    "results/behavioral_causal_audit/qwen30b_dense_cycle_numeric_diagnostic_seed823.json": "94abcd589fe80ca28a508d8eed4f5491ab6c90078a62a716635e3f39a0fdd2eb",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def source_ids(path: Path) -> set[str]:
    return {
        json.loads(line)["source_id"]
        for line in path.read_text().splitlines()
        if line.strip()
    }


def method_family(method: str) -> str | None:
    if method.startswith("foba"):
        return "foba_plus_svd"
    if method.startswith("omp64"):
        return "omp_plus_svd"
    if method.startswith("omp_"):
        return "direct_omp"
    if method in {"top_svd", "gradient_rank"}:
        return method
    return None


def validate_campaign(name: str, config: dict) -> dict:
    stem = config["stem"]
    protocol = ROOT / config["protocol"]
    development_data = ROOT / config["development"]
    confirmation_data = ROOT / config["confirmation"]
    development_summary_path = RESULTS / f"{stem}_development_summary.json"
    confirmation_summary_path = RESULTS / f"{stem}_confirmation_summary.json"
    development_summary = read_json(development_summary_path)
    confirmation_summary = read_json(confirmation_summary_path)

    if source_ids(development_data) & source_ids(confirmation_data):
        raise RuntimeError(f"{name}: development and confirmation sources overlap")
    confirmation_rows = sum(1 for line in confirmation_data.read_text().splitlines() if line.strip())
    if confirmation_rows != config["confirmation_rows"]:
        raise RuntimeError(f"{name}: confirmation row count changed")

    protocol_hash = sha256(protocol)
    development_hash = sha256(development_data)
    confirmation_hash = sha256(confirmation_data)
    developments = {item["training_seed"]: item for item in development_summary["developments"]}
    confirmations = {item["training_seed"]: item for item in confirmation_summary["confirmations"]}
    if tuple(sorted(developments)) != config["seeds"] or tuple(sorted(confirmations)) != config["seeds"]:
        raise RuntimeError(f"{name}: retained seed set changed")
    if development_summary.get("confirmation_opened") is not False:
        raise RuntimeError(f"{name}: development ledger says confirmation was opened")
    if confirmation_summary.get("confirmation_opened") is not True:
        raise RuntimeError(f"{name}: confirmation ledger is not open")

    pooled_raw: dict[str, int] = {}
    pooled_feasible: dict[str, int] = {}
    development_objective_winners: dict[str, int] = {}
    seed_reports = {}
    for seed in config["seeds"]:
        development = developments[seed]
        confirmation = confirmations[seed]
        if development["confirmation_mounted_during_development"] is not False:
            raise RuntimeError(f"{name}/{seed}: confirmation was mounted during development")
        for stage, item, data_hash in (
            ("development", development, development_hash),
            ("confirmation", confirmation, confirmation_hash),
        ):
            if item["stage"] != stage:
                raise RuntimeError(f"{name}/{seed}: wrong stage label")
            if item["training_seed"] != seed:
                raise RuntimeError(f"{name}/{seed}: wrong training seed")
            if item["model_revision"] != config["model_revision"]:
                raise RuntimeError(f"{name}/{seed}: model revision changed")
            if item["parameters"] != config["parameters"]:
                raise RuntimeError(f"{name}/{seed}: parameter count changed")
            if item["protocol_sha256"] != protocol_hash or item["evaluation_data_sha256"] != data_hash:
                raise RuntimeError(f"{name}/{seed}: protocol or data hash mismatch")
            if item["dictionary"]["atoms"] != config["atoms"]:
                raise RuntimeError(f"{name}/{seed}: dictionary size changed")
            if stage == "development" and not item["dense_cycle_pass"]:
                raise RuntimeError(f"{name}/{seed}: development endpoint cycle failed")

        seed_file = RESULTS / f"{stem}_confirmation_seed{seed}.json"
        if read_json(seed_file) != confirmation:
            raise RuntimeError(f"{name}/{seed}: summary and seed artifact differ")
        methods = confirmation["method_records"]
        if confirmation["primary_method"] != config["primary"]:
            raise RuntimeError(f"{name}/{seed}: primary method changed")
        for method, support in development["selection"]["methods"].items():
            if len(support) != config["budget"] or len(set(support)) != config["budget"]:
                raise RuntimeError(f"{name}/{seed}/{method}: support budget changed")
        objective_winner = min(
            development["selection"]["weighted_objectives"],
            key=development["selection"]["weighted_objectives"].get,
        )
        development_objective_winners[objective_winner] = (
            development_objective_winners.get(objective_winner, 0) + 1
        )
        for method, record in methods.items():
            pooled_raw[method] = pooled_raw.get(method, 0) + record["bidirectional_count"]
            pooled_feasible[method] = pooled_feasible.get(method, 0) + (
                record["bidirectional_count"] if record["feasible"] else 0
            )

        primary = methods[config["primary"]]
        behavioral_pass = bool(
            primary["feasible"] and primary["bidirectional_count"] >= 8
        )
        computed_pass = bool(confirmation["dense_cycle_pass"] and behavioral_pass)
        if computed_pass != confirmation["primary_pass"]:
            raise RuntimeError(f"{name}/{seed}: primary pass flag is inconsistent")
        randomization = confirmation["randomization"]
        if randomization["selected_score"] != (
            primary["bidirectional_count"] if primary["feasible"] else 0
        ):
            raise RuntimeError(f"{name}/{seed}: selected randomization score changed")
        if randomization["selected_score"]:
            if randomization["supports"] != 999 or len(randomization["records"]) != 999:
                raise RuntimeError(f"{name}/{seed}: random-support denominator changed")
            at_least = sum(
                row["score"] >= randomization["selected_score"]
                for row in randomization["records"]
            )
            empirical_p = (1 + at_least) / 1000
        else:
            # Every random score is nonnegative, so p=1 is known exactly without
            # spending 999 full model evaluations on a failed selected support.
            at_least = 0
            empirical_p = 1.0
            if randomization["supports"] != 0 or randomization["records"]:
                raise RuntimeError(f"{name}/{seed}: zero-score shortcut changed")
        if randomization["random_at_least_selected"] != at_least or randomization["empirical_p"] != empirical_p:
            raise RuntimeError(f"{name}/{seed}: randomization arithmetic changed")
        seed_reports[str(seed)] = {
            "primary_pass": computed_pass,
            "behavioral_pass": behavioral_pass,
            "dense_cycle_pass": confirmation["dense_cycle_pass"],
            "primary_bidirectional": primary["bidirectional_count"],
            "primary_feasible": primary["feasible"],
            "protected_minimum": min(
                primary["inserted_protected_minimum"],
                primary["ablated_protected_minimum"],
            ),
            "pair_damage": primary["insertion_pair_damage"] + primary["ablation_pair_damage"],
            "random_p": empirical_p,
        }

    if confirmation_summary["pooled_bidirectional_by_method"] != pooled_raw:
        raise RuntimeError(f"{name}: stored pooled method totals changed")
    all_pass = all(item["primary_pass"] for item in seed_reports.values())
    all_behavioral_pass = all(item["behavioral_pass"] for item in seed_reports.values())
    if confirmation_summary["all_primary_seeds_pass"] != all_pass:
        raise RuntimeError(f"{name}: campaign pass flag is inconsistent")

    sealed_paths = [development_summary_path, confirmation_summary_path] + [
        RESULTS / f"{stem}_confirmation_seed{seed}.json" for seed in config["seeds"]
    ]
    observed_seals = {str(path.relative_to(ROOT)): sha256(path) for path in sealed_paths}
    expected_seals = {path: ARTIFACT_HASHES[path] for path in observed_seals}
    if observed_seals != expected_seals:
        raise RuntimeError(f"{name}: sealed result artifact changed")
    return {
        "evidence_class": config["evidence_class"],
        "all_seeds_pass": all_pass,
        "all_behavioral_seeds_pass": all_behavioral_pass,
        "passed_seeds": sum(item["primary_pass"] for item in seed_reports.values()),
        "behavioral_passed_seeds": sum(
            item["behavioral_pass"] for item in seed_reports.values()
        ),
        "total_seeds": len(config["seeds"]),
        "primary_pooled_raw": pooled_raw[config["primary"]],
        "primary_pooled_feasible": pooled_feasible[config["primary"]],
        "pooled_raw_by_method": pooled_raw,
        "pooled_feasible_by_method": pooled_feasible,
        "development_objective_winners": development_objective_winners,
        "seeds": seed_reports,
        "artifact_hashes": observed_seals,
    }


def validate() -> dict:
    campaigns = {name: validate_campaign(name, config) for name, config in CAMPAIGNS.items()}
    selector_comparison = {
        family: {"proxy_wins": 0, "raw_bidirectional": 0, "protected_feasible": 0}
        for family in (
            "foba_plus_svd", "omp_plus_svd", "top_svd", "gradient_rank", "direct_omp"
        )
    }
    for campaign in campaigns.values():
        for method, count in campaign["development_objective_winners"].items():
            family = method_family(method)
            if family is not None:
                selector_comparison[family]["proxy_wins"] += count
        for method, count in campaign["pooled_raw_by_method"].items():
            family = method_family(method)
            if family is not None:
                selector_comparison[family]["raw_bidirectional"] += count
                selector_comparison[family]["protected_feasible"] += (
                    campaign["pooled_feasible_by_method"][method]
                )
    expected_selector_comparison = {
        "foba_plus_svd": {"proxy_wins": 0, "raw_bidirectional": 121, "protected_feasible": 108},
        "omp_plus_svd": {"proxy_wins": 0, "raw_bidirectional": 121, "protected_feasible": 108},
        "top_svd": {"proxy_wins": 0, "raw_bidirectional": 119, "protected_feasible": 108},
        "gradient_rank": {"proxy_wins": 0, "raw_bidirectional": 30, "protected_feasible": 30},
        "direct_omp": {"proxy_wins": 9, "raw_bidirectional": 0, "protected_feasible": 0},
    }
    if selector_comparison != expected_selector_comparison:
        raise RuntimeError("pooled selector comparison changed")
    diagnostic_path = RESULTS / "qwen30b_dense_cycle_numeric_diagnostic_summary.json"
    diagnostic = read_json(diagnostic_path)
    observed_diagnostic_hashes = {
        path: sha256(ROOT / path) for path in DIAGNOSTIC_HASHES
    }
    if observed_diagnostic_hashes != DIAGNOSTIC_HASHES:
        raise RuntimeError("Qwen numeric diagnostic artifact changed")
    if diagnostic["status"] != "float32_unmerged_dense_cycle_pass_all_seeds":
        raise RuntimeError("Qwen numeric diagnostic no longer passes all seeds")
    for item in diagnostic["results"]:
        if (
            item["insertion"]["prediction_agreement"] != 1.0
            or item["ablation"]["prediction_agreement"] != 1.0
        ):
            raise RuntimeError("Qwen float32 endpoint cycle is incomplete")
    return {
        "status": "paper_causal_campaigns_validated",
        "campaigns": campaigns,
        "selector_comparison": selector_comparison,
        "qwen_numeric_diagnostic": {
            "evidence_class": "post_hoc_numeric_diagnostic",
            "all_seeds_float32_cycle_pass": True,
            "maximum_relative_reconstruction_error": max(
                item["maximum_relative_reconstruction_error"]
                for item in diagnostic["results"]
            ),
            "artifact_hashes": observed_diagnostic_hashes,
        },
        "claims": {
            "cross_family_qwen30b_protocol_pass": campaigns["qwen30b"]["all_seeds_pass"],
            "cross_family_qwen30b_behavioral_pass": campaigns["qwen30b"]["all_behavioral_seeds_pass"],
            "fresh_mistral24b_all_seed_replication_pass": campaigns["fresh_mistral24b"]["all_seeds_pass"],
            "exploratory_second_behavior_protocol_pass": campaigns["metadata_abstention"]["all_seeds_pass"],
            "exploratory_second_behavior_behavioral_pass": campaigns["metadata_abstention"]["all_behavioral_seeds_pass"],
            "direct_omp_pooled_bidirectional": {
                name: next(
                    value
                    for method, value in campaign["pooled_raw_by_method"].items()
                    if method.startswith("omp_")
                )
                for name, campaign in campaigns.items()
            },
        },
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
