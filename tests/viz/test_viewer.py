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
    argv = [str(CONFIGS / "mini_deadend.json"), "--vehicles", "4", "--rate", "2", "--hours", "0.1",
            "--screenshot", str(out), "--at", "300"]
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
