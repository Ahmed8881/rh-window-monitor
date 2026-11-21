import pytest

from rhw import config, data


@pytest.fixture(scope="module")
def units():
    if config.UNITS_PATH.exists():
        return data.load_units()
    if data.RHBENCH.exists() and data.TRACE.exists():
        return data.load_all()
    pytest.skip("no data; run scripts/download.py first")


def test_labels_and_turns(units):
    for u in units:
        assert u["label"] in (0, 1)
        assert u["n_turns"] >= 1 and len(u["turns"]) == u["n_turns"]
        assert 1 <= u["n_task"] <= u["n_turns"]


def test_ids_unique(units):
    ids = [u["id"] for u in units]
    assert len(ids) == len(set(ids))


def test_every_source_has_both_classes(units):
    for src in {u["source"] for u in units}:
        labels = {u["label"] for u in units if u["source"] == src}
        assert labels == {0, 1}, src


def test_control_is_single_turn(units):
    for u in units:
        if u["source"] == "control":
            assert [t["role"] for t in u["turns"]] == ["user", "assistant"]


def test_sample_split(units):
    s0, s1 = data.sample_split(units, 0), data.sample_split(units, 0)
    assert s0 == s1  # deterministic per seed
    pos = {u["id"] for u in units if u["label"] == 1}
    assert pos <= s0  # all positives kept
    for src in {u["source"] for u in units}:
        n_pos = sum(1 for u in units if u["source"] == src and u["label"] == 1)
        n_neg = sum(1 for u in units if u["source"] == src and u["label"] == 0 and u["id"] in s0)
        assert n_neg <= min(config.NEG_RATIO * n_pos, config.NEG_CAP)
