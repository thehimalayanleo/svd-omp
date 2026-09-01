#!/usr/bin/env python3
"""Recompute the frozen natural-regression screen decision from item-level scores."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "data/behavior_audit/mistral24b_natural_regression_screen.json"
RESULT_SHA256 = "7f0d3cf253c5b4e645a614b06bec3df8360b5e455bc3ce2b7cb5d742ce8ac579"


def validate() -> dict:
    if hashlib.sha256(RESULT.read_bytes()).hexdigest() != RESULT_SHA256:
        raise RuntimeError("natural-regression result hash mismatch")
    result = json.loads(RESULT.read_text())
    base, post = defaultdict(dict), defaultdict(dict)
    for row in result["base_scores"]:
        base[row["source_id"]][row["family"]] = row
    for row in result["post_scores"]:
        post[row["source_id"]][row["family"]] = row
    families = tuple(result["families"])
    margin = result["minimum_margin"]
    if len(base) != 400 or set(base) != set(post):
        raise RuntimeError("screen does not contain 400 matched sources")
    if any(set(values) != set(families) for values in (*base.values(), *post.values())):
        raise RuntimeError("screen source is missing a required family")

    qualified = []
    base_all = post_protected_all = post_regression = 0
    family_counts = {"base": {}, "post": {}}
    for family in families:
        family_counts["base"][family] = sum(
            values[family]["prediction"] == values[family]["desired"]
            and values[family]["desired_margin"] >= margin for values in base.values()
        )
        family_counts["post"][family] = sum(
            values[family]["prediction"] == values[family]["desired"]
            and values[family]["desired_margin"] >= margin for values in post.values()
        )
    for source in sorted(base):
        base_pass = all(
            base[source][family]["prediction"] == base[source][family]["desired"]
            and base[source][family]["desired_margin"] >= margin
            for family in families
        )
        protected_pass = all(
            post[source][family]["prediction"] == post[source][family]["desired"]
            and post[source][family]["desired_margin"] >= margin
            for family in families if family != "marker_target"
        )
        target = post[source]["marker_target"]
        regression_pass = (
            target["prediction"] == target["regression"]
            and target["regression_margin"] >= margin
        )
        base_all += base_pass
        post_protected_all += protected_pass
        post_regression += regression_pass
        if base_pass and protected_pass and regression_pass:
            qualified.append(source)

    expected = {
        "base_all": 0,
        "post_protected_all": 0,
        "post_regression": 11,
        "qualified": 0,
        "base_quoted_a": 0,
        "post_quoted_a": 0,
    }
    observed = {
        "base_all": base_all,
        "post_protected_all": post_protected_all,
        "post_regression": post_regression,
        "qualified": len(qualified),
        "base_quoted_a": family_counts["base"]["quoted_a"],
        "post_quoted_a": family_counts["post"]["quoted_a"],
    }
    if observed != expected:
        raise RuntimeError(f"unexpected recomputed screen counts: {observed}")
    if qualified != result["qualified_source_ids"]:
        raise RuntimeError("stored qualified IDs differ from recomputation")
    if result["promotion_gate_pass"] or result["status"] != "natural_regression_screen_negative":
        raise RuntimeError("negative promotion decision was not preserved")
    return {
        "status": "validated_negative_natural_regression_screen",
        "result_sha256": RESULT_SHA256,
        "counts": observed,
        "family_task_pass_counts": family_counts,
        "promotion_gate_pass": False,
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
