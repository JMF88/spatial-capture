"""Unit tests for stratified_split (pure, torch-free)."""


def _make(tmp_path, counts):
    for name, n in counts.items():
        d = tmp_path / name
        d.mkdir()
        for i in range(n):
            (d / f"{i}.jpg").write_bytes(b"x")
    return tmp_path


def test_split_keeps_all_classes_and_nonempty_val(load_module, tmp_path):
    s = load_module("understanding/classify/splits.py", "splits")
    root = _make(tmp_path, {"a": 10, "b": 8, "c": 4})
    classes, tr, va, te = s.stratified_split(root, seed=0)
    assert classes == ["a", "b", "c"]
    assert tr and va and te
    for split in (tr, va, te):
        assert {ci for _p, ci in split} == {0, 1, 2}   # every class present in every split


def test_split_tiny_class_no_empty_val_no_leakage(load_module, tmp_path):
    s = load_module("understanding/classify/splits.py", "splits")
    root = _make(tmp_path, {"a": 4, "b": 4, "c": 4})
    _classes, tr, va, te = s.stratified_split(root, seed=0)
    assert len(va) >= 3                                # >=1 val per class
    allp = [p for split in (tr, va, te) for p, _ in split]
    assert len(allp) == len(set(allp))                 # no image in two splits


def test_split_single_image_class_goes_to_train(load_module, tmp_path):
    s = load_module("understanding/classify/splits.py", "splits")
    root = _make(tmp_path, {"solo": 1, "b": 6})
    classes, tr, va, te = s.stratified_split(root, seed=0)
    solo = classes.index("solo")
    assert any(ci == solo for _p, ci in tr)
    assert all(ci != solo for _p, ci in va)
    assert all(ci != solo for _p, ci in te)
