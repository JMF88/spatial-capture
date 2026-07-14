"""Unit tests for the eval gate (pure threshold logic)."""


def test_gate_passes_on_good_metrics(load_module):
    g = load_module("pipeline/gate.py", "gate")
    good = {"reconstruction": {"psnr": 27, "ssim": 0.86},
            "classifier": {"macro_f1": 0.83}, "ocr": {"cer": 0.09}}
    passed, _ = g.check(good)
    assert passed


def test_gate_fails_on_low_psnr(load_module):
    g = load_module("pipeline/gate.py", "gate")
    bad = {"reconstruction": {"psnr": 15, "ssim": 0.86},
           "classifier": {"macro_f1": 0.83}, "ocr": {"cer": 0.09}}
    passed, results = g.check(bad)
    assert not passed
    fails = [r[0] for r in results if r[4] == "FAIL"]
    assert "reconstruction.psnr" in fails


def test_gate_missing_metric_is_not_hard_fail(load_module):
    g = load_module("pipeline/gate.py", "gate")
    passed, results = g.check({"classifier": {"macro_f1": 0.9}})
    assert passed
    assert "missing" in [r[4] for r in results]
