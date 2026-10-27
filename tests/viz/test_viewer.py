import math
import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = pytest.importorskip("pygame")

from agvtb.sidings.choice import candidates_of, parse_siding_choice  # noqa: E402
from agvtb.sim.simulator import SimConfig, Simulation  # noqa: E402
from agvtb.tools.view_sim import main  # noqa: E402
from agvtb.viz.viewer import Viewer  # noqa: E402
from agvtb.world.loader import load_world  # noqa: E402
from helpers import CONFIGS  # noqa: E402


def make(measure=900.0):
    lw = load_world(CONFIGS / "mini_deadend.json")
    return Simulation(lw, candidates_of(lw, parse_siding_choice("grid-bridge-s00")),
                      SimConfig(4, 2.0, 0, 0.0, measure, 0.5))


def ticking(step):
    t = [0.0]

    def clock():
        t[0] += step
        return t[0]
    return clock


def test_frames_advance_the_simulation_by_wall_time_times_speed():
    v = Viewer(make(), speed=10.0)
    assert v.run(max_frames=5, clock=ticking(0.5)) == "frames" and v.sim.now == pytest.approx(20.0)


def test_keys_pause_and_change_speed():
    v = Viewer(make(), speed=10.0)
    for key in (pygame.K_SPACE, pygame.K_PLUS, pygame.K_EQUALS):  # '+' on a Korean or US keyboard is K_EQUALS
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
    v.run(max_frames=3, clock=ticking(1.0))
    assert v.paused and v.speed == 100.0 and v.sim.now == 0.0
    for key in (pygame.K_MINUS, pygame.K_PERIOD):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
    v.run(max_frames=2, clock=ticking(1.0))
    assert v.speed == 30.0 and v.sim.now == pytest.approx(0.5)  # one step of dt while paused


def test_viewer_ends_on_quit():  # Review Focus 4
    v = Viewer(make())
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    assert v.run(max_frames=100, clock=ticking(0.1)) == "quit"


def test_viewer_stops_with_the_run_and_holds_the_last_frame():  # Review Focus 4, plan review I4
    v = Viewer(make(measure=60.0), speed=100.0)
    assert v.run(max_frames=100, clock=ticking(1.0)) == "end" and v.sim.now == 60.0
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    assert v.hold() == "quit"


def test_cli_screenshot(tmp_path, capsys):
    out = tmp_path / "shot.png"
    argv = [str(CONFIGS / "mini_deadend.json"), "--vehicles", "4", "--rate", "2", "--screenshot", str(out), "--at", "300"]
    assert main(argv) == 0 and out.stat().st_size > 1000
    shown = capsys.readouterr().out
    assert "completed" in shown and "screenshot at t 300.0 s" in shown


@pytest.mark.parametrize("extra", [["--vehicles", "0", "--rate", "1"], ["--vehicles", "2", "--rate", "1", "--speed", "0"],
                                   ["--vehicles", "2", "--rate", "1", "--size", "10x10x10"],
                                   ["--vehicles", "2", "--rate", "1", "--size", "100x800"],
                                   ["--vehicles", "2", "--rate", "1", "--screenshot", "x.png", "--at", "-1"]])
def test_cli_usage_errors(extra):
    with pytest.raises(SystemExit) as exc:
        main([str(CONFIGS / "mini_deadend.json")] + extra)
    assert exc.value.code == 2


def test_a_vehicle_in_a_wait_cycle_stands_out_from_a_blocked_one():  # final review I1
    from agvtb.viz.scene import Scene, VehicleView
    from agvtb.viz.viewer import CYCLE_COLOR, KIND_COLOR

    v = Viewer(make())
    x, y = v.sim.world.nodes[v.sim.sidings[0]]
    v._draw(Scene(0.0, (VehicleView(0, x, y, 0.0, "blocked", "I", True),), (), ()))
    poly = v._vehicle_polygon(x, y, 0.0)
    mid = (round((poly[1][0] + poly[2][0]) / 2), round((poly[1][1] + poly[2][1]) / 2))  # middle of the right side
    assert CYCLE_COLOR != KIND_COLOR["blocked"] and tuple(v.screen.get_at(mid))[:3] == CYCLE_COLOR


def test_vehicles_are_at_least_eight_pixels_on_a_big_map():  # final review M4 (re-graded Important)
    v = Viewer(make(), size=(200, 200))
    poly = v._vehicle_polygon(0.0, 0.0, 0.0)
    side = ((poly[0][0] - poly[1][0]) ** 2 + (poly[0][1] - poly[1][1]) ** 2) ** 0.5
    assert side >= 8.0


def test_cli_refuses_a_folder_as_screenshot_and_a_live_run_without_a_display(tmp_path):  # final review M1, M2
    base = [str(CONFIGS / "mini_deadend.json"), "--vehicles", "2", "--rate", "1"]
    with pytest.raises(SystemExit) as exc:
        main(base + ["--screenshot", str(tmp_path), "--at", "10"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(base)  # this module runs with SDL_VIDEODRIVER=dummy: a live window could never be seen or closed
    assert exc.value.code == 2


def test_cli_runs_m6_with_its_rest_rule_and_fast_path(tmp_path, monkeypatch):  # user 2026-10-27: M6b in the viewer
    import agvtb.tools.view_sim as view_sim
    made, real = [], view_sim.Simulation

    def record(lw, candidates, cfg):
        made.append((tuple(candidates), cfg))
        return real(lw, candidates, cfg)
    monkeypatch.setattr(view_sim, "Simulation", record)
    out = tmp_path / "shot.png"
    argv = [str(CONFIGS / "mini_deadend.json"), "--vehicles", "4", "--rate", "2", "--planner", "m6", "--rest-rule", "all",
            "--no-fast-path", "--screenshot", str(out), "--at", "300"]
    assert main(argv) == 0 and out.stat().st_size > 1000
    candidates, cfg = made[0]
    assert candidates == () and (cfg.planner, cfg.rest_rule, cfg.fast_path) == ("m6", "all", False)


@pytest.mark.parametrize("extra, why", [(["--rest-rule", "branch"], "only --planner m6 has a rest rule"),
                                        (["--no-fast-path"], "only --planner m6 has a fast path"),
                                        (["--planner", "m6", "--sidings", "grid-bridge-s00"], "not on sidings")])
def test_cli_m6_options_need_the_m6_planner(extra, why, capsys):  # the words of run_sim
    with pytest.raises(SystemExit) as exc:
        main([str(CONFIGS / "mini_deadend.json"), "--vehicles", "2", "--rate", "1"] + extra)
    assert exc.value.code == 2 and why in capsys.readouterr().err


def test_an_m6_run_draws_its_rest_cells():  # user 2026-10-27; an sl run draws none
    from agvtb.viz.viewer import REST
    lw = load_world(CONFIGS / "mini_deadend.json")
    v = Viewer(Simulation(lw, (), SimConfig(4, 2.0, 0, 0.0, 900.0, 0.5, planner="m6", rest_rule="branch")))
    w, h, r = *v.size, max(3, int(0.3 * v.scale))
    for node in v.sim.m6.rest_nodes:
        cx, cy = (round(c) for c in v.to_screen(*lw.world.nodes[node]))
        box = [(x, y) for x in range(cx - r - 1, cx + r + 2) for y in range(cy - r - 1, cy + r + 2) if 0 <= x < w and 0 <= y < h]
        assert any(tuple(v.background.get_at(p))[:3] == REST for p in box), node
    assert pygame.mask.from_threshold(Viewer(make()).background, REST, (1, 1, 1, 255)).count() == 0


def press(*keys):
    for key in keys:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))


def fast_frames(v):
    """Draw frames until the fast run is over (a frame computes about FAST_FRAME_S of wall time)."""
    v.run(max_frames=1, clock=ticking(1.0))
    for _ in range(1000):
        if v.fast_target is None:
            return
        v.run(max_frames=1, clock=ticking(1.0))
    raise AssertionError("the fast run did not end")


def test_settings_are_typed_in():  # user 2026-10-22: number entry for speed, warm-up and sim time
    v = Viewer(make())
    press(pygame.K_s, pygame.K_2, pygame.K_5, pygame.K_RETURN, pygame.K_w, pygame.K_6, pygame.K_0, pygame.K_0,
          pygame.K_RETURN, pygame.K_t, pygame.K_1, pygame.K_2, pygame.K_0, pygame.K_0, pygame.K_KP_ENTER)
    assert v.run(max_frames=1, clock=ticking(1.0)) == "frames"
    assert (v.speed, v.warmup, v.sim_time) == (25.0, 600.0, 1200.0)


def test_a_bad_or_cancelled_entry_keeps_the_old_value():
    v = Viewer(make())
    press(pygame.K_s, pygame.K_0, pygame.K_RETURN, pygame.K_s, pygame.K_5, pygame.K_ESCAPE)
    assert v.run(max_frames=1, clock=ticking(1.0)) == "frames"  # Esc while typing cancels, it does not quit
    assert v.speed == 10.0 and v.editing is None


def test_fast_mode_runs_to_warm_up_plus_sim_time_then_goes_on_at_speed():  # user 2026-10-22
    v = Viewer(make(math.inf), speed=10.0, warmup=60.0, sim_time=240.0)
    press(pygame.K_f)
    fast_frames(v)
    assert v.sim.now == 300.0 and v.fast_target is None
    v.run(max_frames=3, clock=ticking(1.0))
    assert v.sim.now == 320.0  # the run does not end there: it goes on at x10


def test_fast_mode_does_not_change_the_events():
    v = Viewer(make(math.inf), warmup=0.0, sim_time=600.0)
    press(pygame.K_f)
    fast_frames(v)
    other = make(math.inf)
    other.advance_to(600.0)
    assert v.sim.result().events_sha256 == other.result().events_sha256


def test_kpi_lines_after_the_warm_up():  # user 2026-10-22: KPI of [warm-up end, now], as DAIM
    from agvtb.viz.scene import build_scene

    sim = make(math.inf)
    assert "warm-up: 100 s left" in build_scene(sim, 10.0, False, warmup=100.0).hud
    sim.advance_to(700.0)
    k = sim.kpi(100.0, 700.0)
    lines = build_scene(sim, 10.0, False, warmup=100.0).hud
    assert f"throughput {k['throughput_per_h']:.1f}/h  utilization {k['utilization']:.2f}" in lines[2]
    assert f"cycle {k['cycle_s']:.1f} s  delivery {k['deliver_s']:.1f} s  done {k['completed']}" in lines[3]


def costly(sim, per_sim_s=0.01, tick=0.001):
    """A clock whose wall time also grows with the simulated time computed: computing costs time."""
    calls = [0]

    def clock():
        calls[0] += 1
        return calls[0] * tick + sim.now * per_sim_s
    return clock


def test_no_jump_after_a_fast_run():  # final review I1
    v = Viewer(make(math.inf), speed=10.0, warmup=0.0, sim_time=300.0)
    seen = []
    v.draw = lambda *a: seen.append(v.sim.now)
    press(pygame.K_f)
    v.run(max_frames=60, clock=costly(v.sim))
    i = seen.index(300.0)
    assert seen[i + 1] - 300.0 < 1.0  # the compute time of the fast run is not paced afterwards


def test_a_fast_run_keeps_the_window_answering():  # final review I2
    v = Viewer(make(math.inf), warmup=0.0, sim_time=3600.0)
    press(pygame.K_f)
    v.run(max_frames=1, clock=costly(v.sim))
    assert 0.0 < v.sim.now <= 20.0 and v.fast_target == 3600.0  # a frame computes about 0.1 s, then draws


def test_space_pauses_a_fast_run_and_f_again_stops_it():  # final review minor, re-graded Important
    v = Viewer(make(math.inf), warmup=0.0, sim_time=3600.0)
    press(pygame.K_SPACE, pygame.K_f)  # F while paused starts the fast run
    v.run(max_frames=1, clock=costly(v.sim))
    t = v.sim.now
    assert t > 0.0 and not v.paused
    press(pygame.K_SPACE)
    v.run(max_frames=3, clock=costly(v.sim))
    assert v.sim.now == t and v.fast_target == 3600.0
    press(pygame.K_SPACE, pygame.K_f)
    v.run(max_frames=1, clock=costly(v.sim))
    assert v.fast_target is None and not v.paused


def test_the_window_closes_before_the_summary_is_computed(tmp_path, monkeypatch):  # final review I3
    seen = []
    original = Simulation.result

    def spy(self, *args, **kwargs):
        seen.append(pygame.display.get_init())
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Simulation, "result", spy)
    argv = [str(CONFIGS / "mini_deadend.json"), "--vehicles", "4", "--rate", "2", "--screenshot",
            str(tmp_path / "shot.png"), "--at", "60"]
    assert main(argv) == 0 and seen == [False]


def test_e_stops_the_run_and_e_again_resumes_it():  # user 2026-10-22: (다) a stop key apart from the fast run,
    v = Viewer(make(math.inf), warmup=0.0, sim_time=3600.0)  # then "정지된 상태에서 E를 다시 누르면 가동되게"
    press(pygame.K_s, pygame.K_e, pygame.K_5, pygame.K_RETURN)  # E while a number is typed is not a stop
    assert v.run(max_frames=3, clock=ticking(1.0)) == "frames" and v.speed == 5.0 and not v.halted
    press(pygame.K_f)
    v.run(max_frames=1, clock=costly(v.sim))
    t = v.sim.now
    press(pygame.K_e, pygame.K_SPACE, pygame.K_PERIOD, pygame.K_f)  # stopped: only E and Esc do anything
    assert v.run(max_frames=5, clock=costly(v.sim)) == "frames" and v.sim.now == t and v.halted
    assert "[E] resume" in v._settings_line()
    press(pygame.K_e)
    v.run(max_frames=2, clock=costly(v.sim))
    assert v.sim.now > t and not v.halted and not v.paused and v.fast_target == 3600.0  # the fast run goes on
    press(pygame.K_e, pygame.K_ESCAPE)
    assert v.run(max_frames=5, clock=ticking(1.0)) == "quit"
