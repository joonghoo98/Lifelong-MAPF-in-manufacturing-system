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


def test_a_result_of_another_condition_is_run_again(tmp_path):  # final review I1
    run_batch([mini("a", seed=1)], tmp_path, workers=1)
    out = run_batch([mini("a", seed=2)], tmp_path, workers=1)
    assert out[0]["skipped"] is False and json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))["seed"] == 2


def test_a_skipped_run_keeps_its_result(tmp_path):  # final review I2: a stored "not ok" is not exit 0
    run_batch([mini("a")], tmp_path, workers=1)
    data = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    (tmp_path / "a.json").write_text(json.dumps(data | {"ok": False}), encoding="utf-8")
    assert run_batch([mini("a")], tmp_path, workers=1)[0] == {"name": "a", "exit": 1, "skipped": True}


def test_rates_keep_every_digit_and_the_summary_keeps_to_the_conditions(tmp_path):  # final review minors
    from agvtb.tools.batch import run_argv
    assert run_argv(mini("a", rate=0.1234567), tmp_path / "a.json")[4] == "0.1234567"
    run_batch([mini("a")], tmp_path, workers=1)
    (tmp_path / "conds.json").write_text(json.dumps([mini("a")]), encoding="utf-8")
    assert [r["name"] for r in summary_rows(tmp_path, ["a"])] == ["a"]


def test_a_cut_condition_passes_the_switch_and_resumes_only_its_own(tmp_path):  # M5c; A2 is on by default
    from agvtb.tools.batch import run_argv
    out_json = tmp_path / "a.json"
    assert not {"--tail-cut", "--no-tail-cut", "--cut-margin"} & set(run_argv(mini("a"), out_json))  # old command kept
    assert "--no-tail-cut" in run_argv(mini("a", tail_cut=False), out_json)
    on = run_argv(mini("a", tail_cut=True, cut_margin=1.0), out_json)
    assert "--tail-cut" in on and on[on.index("--cut-margin") + 1] == "1.0"
    run_batch([mini("a", tail_cut=False)], tmp_path, workers=1)
    stored = json.loads(out_json.read_text(encoding="utf-8"))
    assert stored["tail_cut"] is False
    del stored["tail_cut"], stored["cut_margin"]  # a result from before A2 counts as off
    out_json.write_text(json.dumps(stored), encoding="utf-8")
    assert run_batch([mini("a", tail_cut=False)], tmp_path, workers=1)[0]["skipped"] is True
    assert run_batch([mini("a")], tmp_path, workers=1)[0]["skipped"] is False  # no field: run_sim's default, on
    assert json.loads(out_json.read_text(encoding="utf-8"))["tail_cut"] is True


def test_an_m6a_condition_turns_the_fast_path_off_and_resumes_only_its_own(tmp_path):  # M6c plan: the M6a arm
    from agvtb.tools.batch import run_argv
    out_json = tmp_path / "a.json"
    m6a = mini("a", planner="m6", rest_rule="branch", fast_path=False)
    assert "--no-fast-path" not in run_argv(mini("a", planner="m6", rest_rule="branch"), out_json)
    assert "--no-fast-path" in run_argv(m6a, out_json)
    run_batch([m6a], tmp_path, workers=1)
    assert json.loads(out_json.read_text(encoding="utf-8"))["fast_path"] is False
    assert run_batch([m6a], tmp_path, workers=1)[0]["skipped"] is True
    assert run_batch([mini("a", planner="m6", rest_rule="branch")], tmp_path, workers=1)[0]["skipped"] is False
    stored = json.loads(out_json.read_text(encoding="utf-8"))
    del stored["fast_path"]  # a result from before M6c: its stats tell whether the fast path ran (R30) or not (R29)
    out_json.write_text(json.dumps(stored), encoding="utf-8")
    assert run_batch([mini("a", planner="m6", rest_rule="branch")], tmp_path, workers=1)[0]["skipped"] is True
    del stored["stats"]["m6_fast_ok"]
    out_json.write_text(json.dumps(stored), encoding="utf-8")
    assert run_batch([m6a], tmp_path, workers=1)[0]["skipped"] is True


def test_an_m6_condition_passes_its_planner_and_resumes_only_its_own(tmp_path):
    from agvtb.tools.batch import run_argv
    out_json = tmp_path / "a.json"
    assert "--planner" not in run_argv(mini("a"), out_json)  # the sl command stays as it was
    argv = run_argv(mini("a", planner="m6", rest_rule="branch"), out_json)
    assert argv[argv.index("--planner") + 1] == "m6" and argv[argv.index("--rest-rule") + 1] == "branch"
    run_batch([mini("a", planner="m6", rest_rule="branch")], tmp_path, workers=1)
    stored = json.loads(out_json.read_text(encoding="utf-8"))
    assert (stored["planner"], stored["rest_rule"], stored["tail_cut"]) == ("m6", "branch", False)
    assert run_batch([mini("a", planner="m6", rest_rule="branch")], tmp_path, workers=1)[0]["skipped"] is True
    assert run_batch([mini("a", planner="m6")], tmp_path, workers=1)[0]["skipped"] is False  # no-cut: another run
    assert run_batch([mini("a")], tmp_path, workers=1)[0]["skipped"] is False  # an sl condition: another run
    stored = json.loads(out_json.read_text(encoding="utf-8"))
    del stored["planner"], stored["rest_rule"]  # a result from before M6 ran sl
    out_json.write_text(json.dumps(stored), encoding="utf-8")
    assert run_batch([mini("a")], tmp_path, workers=1)[0]["skipped"] is True
