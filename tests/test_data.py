"""Dataset tests: no train/test leakage, labels aligned with their construction."""
from emoprobe.datasets import build_all, check_leakage, label_alignment_ok


def test_no_leakage():
    check_leakage(build_all())


def test_label_alignment():
    assert label_alignment_ok(build_all())


def test_deterministic():
    assert build_all() == build_all()


def test_d1_pairs_share_task_and_paraphrase_slot():
    ex = [e for e in build_all() if e["dataset"] == "d1"]
    by_item = {}
    for e in ex:
        by_item.setdefault(e["item"], []).append(e)
    for item, rows in by_item.items():
        assert sorted(r["label"] for r in rows) == [0, 1]
        assert len({r["user"] for r in rows}) == 1
        assert len({r["para"] % 5 for r in rows}) == 1


def test_balanced_labels():
    ex = build_all()
    for ds in ("d1", "d2", "d1neg"):
        labs = [e["label"] for e in ex if e["dataset"] == ds]
        assert labs.count(0) == labs.count(1)
