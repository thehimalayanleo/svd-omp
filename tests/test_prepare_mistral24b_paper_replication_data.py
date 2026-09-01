from prepare_mistral24b_paper_replication_data import ALLOCATION, build


def test_fresh_replication_partitions_are_disjoint_and_sized():
    manifest = build()
    development = manifest["selected_sources"]["development"]
    confirmation = manifest["selected_sources"]["confirmation"]
    assert len(development) == sum(ALLOCATION["development"].values()) == 12
    assert len(confirmation) == sum(ALLOCATION["confirmation"].values()) == 16
    assert set(development).isdisjoint(confirmation)
    assert manifest["source_disjoint_from_all_prior_mistral24b_campaigns"] is True
    assert manifest["outputs"]["development"]["rows"] == 96
    assert manifest["outputs"]["confirmation"]["rows"] == 128
