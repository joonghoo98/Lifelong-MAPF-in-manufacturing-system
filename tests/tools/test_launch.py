from agvtb.tools.launch import build_argv, main, runnable_configs
from helpers import CONFIGS, REPO


def feed(*answers):
    it = iter(answers)
    return lambda prompt: next(it)


def test_runnable_configs_are_the_maps_with_a_siding_variant():
    names = [p.stem for p in runnable_configs(CONFIGS)]
    assert names == ["grid_songdo_manual", "KAIROS_AMR", "mini_deadend", "random_32_32_20", "songdo_grid_bi"]


def test_blank_answers_take_the_defaults():
    argv = build_argv(feed("", "", "", ""), lambda *a: None, CONFIGS)
    assert argv == [str(CONFIGS / "mini_deadend.json"), "--vehicles", "8", "--rate", "2", "--seed", "0"]


def test_bad_answers_are_asked_again():
    shown = []
    argv = build_argv(feed("9", "2", "0", "x", "4", "-1", "1.5", "s", "3"), shown.append, CONFIGS)
    assert argv == [str(CONFIGS / "KAIROS_AMR.json"), "--vehicles", "4", "--rate", "1.5", "--seed", "3"]
    assert sum("try again" in line for line in shown) == 5


def test_main_runs_the_viewer_with_the_answers():
    got = []
    assert main(input_fn=feed("", "", "", ""), run=lambda argv: got.append(argv) or 0, root=CONFIGS) == 0
    assert got[0][1:3] == ["--vehicles", "8"]


def test_the_bat_file_runs_the_launcher_from_its_own_folder():
    text = (REPO / "view_sim.bat").read_text(encoding="ascii")
    assert 'cd /d "%~dp0"' in text and '".venv\\Scripts\\python.exe" -m agvtb.tools.launch' in text


def test_ctrl_c_or_end_of_input_ends_quietly(capsys):  # review: no traceback
    def stop(prompt):
        raise KeyboardInterrupt

    def eof(prompt):
        raise EOFError

    assert main(input_fn=stop, run=lambda argv: 0, root=CONFIGS) == 1
    assert main(input_fn=eof, run=lambda argv: 0, root=CONFIGS) == 1
    assert "Traceback" not in capsys.readouterr().out


def test_a_gate_refusal_asks_again(capsys):  # review: too many vehicles for the map
    codes = iter([3, 0])
    got = []
    assert main(input_fn=feed("2", "99", "", "", "", "", "", ""),
                run=lambda argv: got.append(argv) or next(codes), root=CONFIGS) == 0
    assert [a[2] for a in got] == ["99", "8"] and "try fewer vehicles or another map" in capsys.readouterr().out


def test_words_get_a_plain_message():
    shown = []
    build_argv(feed("abc", "", "abc", "", "abc", "", "abc", ""), shown.append, CONFIGS)
    assert len(shown) > 4 and not any("invalid literal" in line or "could not convert" in line for line in shown)


def test_the_keys_shown_include_the_stop_key(capsys):  # user 2026-10-22 (다)
    main(input_fn=feed("", "", "", ""), run=lambda argv: 0, root=CONFIGS)
    assert "E stop" in capsys.readouterr().out
