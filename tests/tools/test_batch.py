import json

import pytest

from agvtb.tools.batch import passed, run_batch, summary_rows, sweep
from helpers import CONFIGS


def mini(name, **kw):
    return {"name": name, "config": str(CONFIGS / "mini_deadend.json"), "vehicles": 4, "rate": 2.0, "seed": 0,
            "warmup": 0.0, "measure": 600.0, "sidings": None, "fromto": None} | kw  # jobs finish by 600 s


def test_a_batch_runs_every_condition_and_resumes(tmp_path):
    out = run_batch([mini("a"), mini("b", seed=1)], tmp_path, workers=2)
    assert [(r["name"], r["exit"], r["skipped"]) for r in out] == [("a", 0, False), ("b", 0, False)]
    rows = summary_rows(tmp_path)
    assert [r["name"] for r in rows] == ["a", "b"] and all(r["passed"] for r in rows)
    before = (tmp_path / "a.json").stat().st_mtime_ns
    assert all(r["skipped"] for r in run_batch([mini("a"), mini("b", seed=1)], tmp_path, workers=2))
    assert (tmp_path / "a.json").stat().st_mtime_ns == before


def test_a_half_written_result_is_run_again(tmp_path):  # Review Focus 1
    (tmp_path / "a.json").write_text("{", encoding="utf-8")
    assert run_batch([mini("a")], tmp_path, workers=1)[0]["skipped"] is False
    json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))


def test_a_refused_run_is_recorded_and_the_batch_goes_on(tmp_path):  # Review Focus 4
    bad = mini("k", config=str(CONFIGS / "KAIROS_AMR.json"), vehicles=13)
    out = run_batch([bad, mini("a")], tmp_path, workers=1)
    assert [r["exit"] for r in out] == [3, 0] and not (tmp_path / "k.json").exists()
    assert "done k exit 3" in (tmp_path / "progress.txt").read_text(encoding="utf-8")


def test_names_must_be_unique(tmp_path):
    with pytest.raises(ValueError):
        run_batch([mini("a"), mini("a")], tmp_path, workers=1)


def test_sweep_makes_every_combination():
    conds = sweep("configs/maps/mini_deadend.json", [2, 4], [1.0, 2.5], [0, 1], 0.0, 600.0, None, None)
    assert len(conds) == 8 and len({c["name"] for c in conds}) == 8
    assert conds[0]["name"] == "mini_deadend_n2_l1_s0" and conds[-1]["name"] == "mini_deadend_n4_l2.5_s1"


def test_passed_is_the_m4c_rule():
    good = {"ok": True, "t_end": 900.0, "warmup_s": 300.0, "measure_s": 600.0, "job_max_s": 100.0, "stats": {}}
    assert passed(good)
    for bad in ({"ok": False}, {"t_end": 899.0}, {"job_max_s": None}, {"job_max_s": 1800.0}, {"stats": {"plan_fail": 1}}):
        assert not passed(good | bad)
