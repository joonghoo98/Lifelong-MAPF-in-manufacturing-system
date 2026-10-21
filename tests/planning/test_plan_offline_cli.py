import json

import pytest

from agvtb.tools.plan_offline import main
from helpers import CONFIGS, DAIM_CONFIGS


def test_cli_plans_mini_with_the_configs_variant(tmp_path, capsys):
    out = tmp_path / "mini.json"
    assert main([str(CONFIGS / "mini_deadend.json"), "--vehicles", "8", "--tasks", "24", "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert (data["sidings_choice"], data["vehicles"], data["tasks"], data["seed"], data["budget"]) == (
        "grid-bridge-s00", 8, 24, 0, 200_000)
    assert data["ok"] is True and data["gate"]["ok"] is True and data["stats"] == {"search_ok": 24}
    assert data["timelines_sha256"] == "c95422ec0f0559160f193023a06867badce2e4010b5b0b9a05ac489d39ed1b4f"
    assert len(data["plans"]) == 24 and data["homes"]["7"] == "G14_6" and data["gate"]["paper_outside"] == []
    shown = capsys.readouterr().out
    assert "24 of 24 tasks planned" in shown and "paper condition (not part of the gate): 0 nodes outside" in shown


def test_cli_gate_refusal_writes_nothing(tmp_path, capsys):  # Review Focus 5
    out = tmp_path / "refused.json"
    assert main([str(CONFIGS / "mini_deadend.json"), "--vehicles", "38", "--tasks", "1", "--out", str(out)]) == 3
    assert not out.exists() and "only 37 usable sidings for 38 vehicles" in capsys.readouterr().out


def test_cli_sidings_override(tmp_path):  # offline_more.out.txt 3.
    out = tmp_path / "strict.json"
    argv = [str(CONFIGS / "mini_deadend.json"), "--vehicles", "2", "--tasks", "2", "--sidings", "strict"]
    assert main(argv + ["--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["sidings_choice"] == "strict"


@pytest.mark.parametrize("argv", [
    [str(DAIM_CONFIGS / "grid_songdo_travelbr.json"), "--vehicles", "2", "--tasks", "2"],  # no choice anywhere
    [str(CONFIGS / "mini_deadend.json"), "--vehicles", "2", "--tasks", "1", "--sidings", "random"],
    [str(CONFIGS / "mini_deadend.json"), "--vehicles", "0", "--tasks", "1"],
    # plan_review_1.md finding 2 (review_checks/cli_errors.out.txt): these ended in a traceback and exit code 1
    [str(CONFIGS / "mini_deadend.json"), "--vehicles", "2", "--tasks", "1", "--budget", "0"],
    [str(CONFIGS / "mini_deadend.json"), "--vehicles", "2", "--tasks", "1", "--interval", "-1"],
    [str(CONFIGS / "mini_deadend.json"), "--vehicles", "2", "--tasks", "1", "--sidings", "recorded"],  # no record
    [str(CONFIGS / "KAIROS_AMR.json"), "--vehicles", "2", "--tasks", "1", "--sidings", "grid-bridge-s00"],
    [str(CONFIGS / "no_such_map.json"), "--vehicles", "2", "--tasks", "1"],
    # plan_review_2.md finding 1: an --out folder that does not exist is seen before any planning
    [str(CONFIGS / "mini_deadend.json"), "--vehicles", "2", "--tasks", "1", "--out", "no_such_dir/x.json"],
    # final review: --out naming a folder, and dispatch times that overflow, also end before planning
    [str(CONFIGS / "mini_deadend.json"), "--vehicles", "2", "--tasks", "1", "--out", str(CONFIGS)],
    [str(CONFIGS / "mini_deadend.json"), "--vehicles", "1", "--tasks", "3", "--interval", "1e308"],
])
def test_cli_usage_errors(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
