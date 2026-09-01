import json

import prepare_mistral24b_multiseed_confirmation_data as frozen


def test_multiseed_partitions_are_balanced_fresh_and_hashed():
    manifest = frozen.build()
    prior_sources = set()
    for path in frozen.PRIOR_MANIFESTS:
        prior = json.loads(path.read_text())
        prior_sources.update(source for values in prior["selected_sources"].values() for source in values)

    selected = manifest["selected_sources"]
    flattened = [source for values in selected.values() for source in values]
    assert len(flattened) == 36
    assert len(flattened) == len(set(flattened))
    assert not set(flattened) & prior_sources
    assert manifest["original_24_source_final_test_opened"] is False

    expected_sources = {"development": 12, "validation": 8, "confirmation": 16}
    for partition, count in expected_sources.items():
        assert manifest["outputs"][partition]["sources"] == count
        assert manifest["outputs"][partition]["rows"] == count * 8
        categories = [source.split(":", 1)[0] for source in selected[partition]]
        per_category = frozen.PER_CATEGORY[partition]
        assert all(categories.count(category) == per_category for category in frozen.CATEGORIES)
        assert frozen.sha256(frozen.OUTPUTS[partition]) == manifest["outputs"][partition]["sha256"]


def test_each_source_has_the_complete_eight_family_factorial():
    frozen.build()
    expected = {
        "ambiguous", "clean_a", "clean_b", "marked_ambiguous",
        "marker_control", "marker_target", "quoted_a", "quoted_b",
    }
    for path in frozen.OUTPUTS.values():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        sources = {row["source_id"] for row in rows}
        for source in sources:
            assert {row["family"] for row in rows if row["source_id"] == source} == expected
