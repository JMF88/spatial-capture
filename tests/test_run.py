"""Unit tests for the orchestrator's pure stage-planning (no execution)."""


def test_plan_full_pipeline(load_module):
    r = load_module("pipeline/run.py", "run_orch")
    cfg = {"scene": "s", "video": "data/s/v.mp4", "workdir": "data/s",
           "sparse": "data/s/sparse/0", "splat": "data/s/s.ply",
           "detect": {"classes": ["book", "lamp"]}, "publish": "docs/viewer/assets"}
    plan = r.plan_stages(cfg, repo="/repo", python_exe="py")
    assert [s["name"] for s in plan] == ["frames", "detect", "fusion", "publish"]
    detect = next(s for s in plan if s["name"] == "detect")
    assert "book,lamp" in detect["argv"]          # classes are comma-joined for detect.py
    publish = next(s for s in plan if s["name"] == "publish")
    assert publish["argv"][0] == "@copy"          # publish is a file-copy pseudo-stage


def test_plan_from_and_only(load_module):
    r = load_module("pipeline/run.py", "run_orch")
    cfg = {"video": "v", "workdir": "w", "sparse": "sp", "splat": "sm", "publish": "pub"}
    from_fusion = [s["name"] for s in r.plan_stages(cfg, "/r", from_stage="fusion")]
    assert from_fusion == ["fusion", "publish"]
    only_detect = [s["name"] for s in r.plan_stages(cfg, "/r", only="detect")]
    assert only_detect == ["detect"]
