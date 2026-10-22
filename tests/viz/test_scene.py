import pytest

from agvtb.sidings.choice import candidates_of, parse_siding_choice
from agvtb.sim.simulator import SimConfig, Simulation
from agvtb.viz.scene import build_scene
from agvtb.world.loader import load_world
from helpers import CONFIGS


@pytest.fixture(scope="module")
def sim():
    lw = load_world(CONFIGS / "mini_deadend.json")
    s = Simulation(lw, candidates_of(lw, parse_siding_choice("grid-bridge-s00")), SimConfig(4, 2.0, 0, 0.0, 900.0, 0.5))
    s.advance_to(300.0)
    return s


def test_scene_lists_every_vehicle_and_port_on_the_map(sim):
    sc = build_scene(sim, 10.0, False)
    xs = [x for x, _ in sim.world.nodes.values()]
    ys = [y for _, y in sim.world.nodes.values()]
    assert [v.vid for v in sc.vehicles] == [0, 1, 2, 3] and len(sc.ports) == len(sim.world.port_nodes)
    assert all(min(xs) - 1e-9 <= v.x <= max(xs) + 1e-9 and min(ys) - 1e-9 <= v.y <= max(ys) + 1e-9 for v in sc.vehicles)
    assert {v.kind for v in sc.vehicles} <= {"move", "rotate", "blocked", "dwell", "rest"}
    assert {v.phase for v in sc.vehicles} <= set("IPLDU") and not any(v.in_cycle for v in sc.vehicles)


def test_hud_lines(sim):
    sc = build_scene(sim, 30.0, True)
    assert sc.hud[0] == "t     300.0 s  x30  paused" and sc.hud[1].startswith("done ")


def test_port_counts_are_the_waiting_jobs():  # plan review 1, minor 1: a time with jobs waiting
    lw = load_world(CONFIGS / "mini_deadend.json")
    busy = Simulation(lw, candidates_of(lw, parse_siding_choice("grid-bridge-s00")), SimConfig(2, 4.0, 0, 0.0, 900.0, 0.5))
    busy.advance_to(150.0)
    sc = build_scene(busy, 10.0, False)
    assert len(busy.mcs.waiting()) > 0 and sum(p.waiting for p in sc.ports) == len(busy.mcs.waiting())
