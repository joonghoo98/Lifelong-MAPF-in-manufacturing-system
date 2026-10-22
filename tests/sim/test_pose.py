import math

import pytest

from agvtb.sim.executor import Executor, world_geometry
from agvtb.sim.kernel import Kernel
from agvtb.sim.pose import pose_of
from agvtb.sim.tapg import Tapg
from helpers import spur_room_world

TURN = [("S,c00,0", 0.0), ("M,c00,c10", 0.0), ("S,c10,0", 1.0), ("R,c10,0,90", 1.0), ("S,c10,90", 3.5),
        ("M,c10,c11", 3.5), ("S,c11,90", 4.5)]
TRI = 4 * math.sqrt(0.5)  # the 1 m move from rest to rest


def started(tmp_path, path, until):
    lw = spur_room_world(tmp_path)
    k, g = Kernel(), Tapg(lw.graph, lw.conflicts)
    g.add_vehicle(1, path[0][0], 0.0)
    ex = Executor(lw.graph, g, k, world_geometry(lw.world), 0.5, 0.5, lambda *a: None)
    g.append(1, path)
    ex.add(1)
    k.run_until(until, dt=0.01)
    return lw, ex


def test_pose_on_a_move_rotation_and_rest(tmp_path):
    lw, ex = started(tmp_path, TURN, 1.0)
    p = pose_of(lw.world, lw.graph, ex, 1, 1.0)
    assert (p.x, p.y, p.heading_deg, p.kind) == (pytest.approx(0.25), pytest.approx(0.0), 0.0, "move")
    lw, ex = started(tmp_path, TURN, TRI + 1.25)
    p = pose_of(lw.world, lw.graph, ex, 1, TRI + 1.25)  # half way through the 90 degree turn
    assert (p.x, p.y, p.heading_deg, p.kind) == (pytest.approx(1.0), pytest.approx(0.0), pytest.approx(45.0), "rotate")


def test_pose_along_a_passing_move(tmp_path):  # Review Focus 2: no jump at the braking point
    path = [("S,c00,0", 0.0), ("M,c00,c10", 0.0), ("S,c10,0", 1.0), ("M,c10,c20", 1.0), ("S,c20,0", 2.0)]
    xs = []
    for t in (1.40, 1.41, 1.42, 1.43):  # around the first move's braking point (sqrt 2 s)
        lw, ex = started(tmp_path, path, t)
        xs.append(pose_of(lw.world, lw.graph, ex, 1, t).x)
    assert xs == sorted(xs) and max(b - a for a, b in zip(xs, xs[1:])) < 0.02
    lw, ex = started(tmp_path, path, 1.9)  # after the braking point the passing profile holds: 0.25 t^2
    assert pose_of(lw.world, lw.graph, ex, 1, 1.9).x == pytest.approx(0.25 * 1.9 ** 2)  # a kept stop profile: 0.7845


def test_pose_of_a_vehicle_at_rest(tmp_path):  # Review Focus 3
    lw, ex = started(tmp_path, TURN, 60.0)
    assert ex.vehicles[1].state == "DONE"
    p = pose_of(lw.world, lw.graph, ex, 1, 60.0)
    assert (p.x, p.y, p.heading_deg, p.kind) == (pytest.approx(1.0), pytest.approx(1.0), 90.0, "rest")
