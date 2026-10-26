import json

import pytest

from agvtb.tools.run_sim import main
from helpers import CONFIGS


def test_cli_runs_mini_and_writes_the_result(tmp_path, capsys):
    out = tmp_path / "r.json"
    argv = [str(CONFIGS / "mini_deadend.json"), "--vehicles", "4", "--rate", "2", "--warmup", "0", "--measure", "600"]
    assert main(argv + ["--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["ok"] is True and data["sidings_choice"] == "grid-bridge-s00" and data["deadlock"]["global"] is False
    assert len(data["events_sha256"]) == 64 and "completed" in capsys.readouterr().out


def test_cli_gate_refusal_writes_nothing(tmp_path):
    out = tmp_path / "r.json"
    assert main([str(CONFIGS / "mini_deadend.json"), "--vehicles", "38", "--rate", "1", "--out", str(out)]) == 3
    assert not out.exists()


@pytest.mark.parametrize("extra", [["--vehicles", "0", "--rate", "1"], ["--vehicles", "2", "--rate", "-1"],
                                   ["--vehicles", "2", "--rate", "1", "--dt", "0"],
                                   ["--vehicles", "2", "--rate", "1", "--sidings", "random"]])
def test_cli_usage_errors(extra):
    with pytest.raises(SystemExit) as exc:
        main([str(CONFIGS / "mini_deadend.json")] + extra)
    assert exc.value.code == 2


def test_out_json_has_the_values_the_st24_comparison_needs(tmp_path):  # M4c
    from agvtb.sidings.choice import candidates_of, parse_siding_choice
    from agvtb.sim.simulator import SimConfig, simulate
    from agvtb.world.config import load_map_config
    from agvtb.world.loader import load_world

    out = tmp_path / "r.json"
    # the warm-up ends after the run's longest job (113.9 s, done at 840.3 s): job_max_s must count warm-up jobs too,
    # as st24 does (final review)
    argv = [str(CONFIGS / "mini_deadend.json"), "--vehicles", "4", "--rate", "2", "--warmup", "841", "--measure", "59",
            "--out", str(out)]
    assert main(argv) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    lw = load_world(CONFIGS / "mini_deadend.json")
    cands = candidates_of(lw, parse_siding_choice(load_map_config(CONFIGS / "mini_deadend.json").sidings))
    r = simulate(lw, cands, SimConfig(4, 2.0, 0, 841.0, 59.0, 0.5))
    longest = max((j for j in r.jobs if j.t_completed is not None), key=lambda j: j.t_completed - j.t_assigned)
    assert longest.t_completed < 841.0  # so a window-only maximum would differ
    assert data["job_max_s"] == longest.t_completed - longest.t_assigned
    assert data["jobs_created"] == len(r.jobs) and data["wall_s"] > 0


def test_run_sim_records_the_switch(tmp_path):  # M5c: A2 on by default, --no-tail-cut turns it off
    base = [str(CONFIGS / "mini_deadend.json"), "--vehicles", "4", "--rate", "2", "--warmup", "0", "--measure", "120"]
    assert main(base + ["--out", str(tmp_path / "on.json")]) in (0, 1)
    assert main(base + ["--no-tail-cut", "--cut-margin", "1", "--out", str(tmp_path / "off.json")]) in (0, 1)
    on, off = (json.loads((tmp_path / f"{n}.json").read_text(encoding="utf-8")) for n in ("on", "off"))
    assert (on["tail_cut"], on["cut_margin"], off["tail_cut"], off["cut_margin"]) == (True, 0.0, False, 1.0)


@pytest.mark.parametrize("margin", ["-1", "nan", "inf"])
def test_cut_margin_must_be_finite_and_not_negative(margin, capsys):
    with pytest.raises(SystemExit):
        main([str(CONFIGS / "mini_deadend.json"), "--vehicles", "4", "--rate", "2", "--cut-margin", margin])
    assert "--cut-margin is out of range" in capsys.readouterr().err


KAIROS = str(CONFIGS / "KAIROS_AMR.json")


def test_cli_runs_m6_and_records_the_mode(tmp_path):
    out = tmp_path / "m6.json"
    code = main([KAIROS, "--vehicles", "20", "--rate", "1", "--warmup", "0", "--measure", "300", "--planner", "m6",
                 "--rest-rule", "branch", "--out", str(out)])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert code in (0, 1)
    assert (data["planner"], data["rest_rule"], data["tail_cut"], data["sidings_choice"]) == ("m6", "branch", False, None)
    assert data["gate"]["ok"] and data["gate"]["rule"] == "branch" and "slow_jobs" in data["m6"]


def test_cli_m6_gate_refusal_writes_nothing(tmp_path):
    out = tmp_path / "x.json"
    argv = [KAIROS, "--vehicles", "27", "--rate", "1", "--planner", "m6", "--rest-rule", "branch", "--out", str(out)]
    assert main(argv) == 3 and not out.exists()


@pytest.mark.parametrize("extra", [["--planner", "m6", "--tail-cut"], ["--planner", "m6", "--sidings", "strict"],
                                   ["--rest-rule", "branch"]])
def test_cli_m6_usage_errors(extra):  # plan decision 5
    with pytest.raises(SystemExit) as exc:
        main([KAIROS, "--vehicles", "4", "--rate", "1"] + extra)
    assert exc.value.code == 2


def test_an_sl_result_names_its_planner(tmp_path):
    out = tmp_path / "sl.json"
    main([str(CONFIGS / "mini_deadend.json"), "--vehicles", "4", "--rate", "2", "--warmup", "0", "--measure", "120",
          "--out", str(out)])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert (data["planner"], data["rest_rule"], data["tail_cut"]) == ("sl", None, True) and "m6" not in data
