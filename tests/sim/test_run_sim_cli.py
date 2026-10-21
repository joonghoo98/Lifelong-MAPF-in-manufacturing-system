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
