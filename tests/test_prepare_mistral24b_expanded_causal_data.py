import json

from prepare_mistral24b_expanded_causal_data import DEV_A, DEV_B, PRIOR, build


def test_expanded_splits_are_fresh_source_disjoint_factorials() -> None:
    manifest = build()
    prior = json.loads(PRIOR.read_text())
    used = {source for values in prior["selected_sources"].values() for source in values}
    rows_a = [json.loads(line) for line in DEV_A.read_text().splitlines() if line]
    rows_b = [json.loads(line) for line in DEV_B.read_text().splitlines() if line]
    sources_a = {row["source_id"] for row in rows_a}
    sources_b = {row["source_id"] for row in rows_b}
    assert len(rows_a) == len(rows_b) == 128
    assert len(sources_a) == len(sources_b) == 16
    assert not sources_a & sources_b
    assert not (sources_a | sources_b) & used
    assert manifest["original_final_test_mounted"] is False


def test_each_source_has_all_eight_behavior_families() -> None:
    build()
    required = {
        "ambiguous", "clean_a", "clean_b", "marked_ambiguous",
        "marker_control", "marker_target", "quoted_a", "quoted_b",
    }
    for path in (DEV_A, DEV_B):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        for source in {row["source_id"] for row in rows}:
            assert {row["family"] for row in rows if row["source_id"] == source} == required
