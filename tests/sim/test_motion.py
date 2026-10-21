import math

import pytest

from agvtb.sim.motion import distance_at, plan_move

INF = math.inf


def flat(plan):
    return [x for frame in plan.keyframes for x in frame]


# research §C1 (DAIM agv_motion.py:33-131); numbers recomputed by hand: 1 m cell, 0.5 m/s, 0.5 m/s^2
def test_one_cell_stop_to_stop_is_three_seconds():
    p = plan_move(0.0, 0.5, 0.5, 0.5, 1.0, stop=True)
    assert (p.duration_s, p.brake_s, p.v_end) == pytest.approx((3.0, 2.0, 0.0))
    assert flat(p) == pytest.approx([0, 0, 0, 1, 0.25, 0.5, 2, 0.75, 0.5, 3, 1, 0])


def test_passing_cells_take_two_n_plus_one_seconds():
    first = plan_move(0.0, 0.5, 0.5, 0.5, 1.0, stop=False)
    middle = plan_move(0.5, 0.5, 0.5, 0.5, 1.0, stop=False)
    last = plan_move(0.5, 0.5, 0.5, 0.5, 1.0, stop=True)
    assert (first.duration_s, first.v_end, first.brake_s) == pytest.approx((2.5, 0.5, 2.5))
    assert (middle.duration_s, last.duration_s, last.brake_s) == pytest.approx((2.0, 2.5, 1.5))
    for n in (2, 3, 5):
        assert first.duration_s + (n - 2) * middle.duration_s + last.duration_s == pytest.approx(2 * n + 1)


def test_triangle_short_and_cannot_stop_cases():
    tri = plan_move(0.0, 0.5, 0.5, 0.5, 0.25, stop=True)  # vp^2 = 0.25 / (1 + 1)
    assert (tri.duration_s, tri.brake_s) == pytest.approx((math.sqrt(2), math.sqrt(0.5)))
    short = plan_move(0.0, 0.5, 0.5, 0.5, 0.16, stop=False)  # never reaches vmax
    assert (short.duration_s, short.v_end) == pytest.approx((0.8, 0.4))
    late = plan_move(0.5, 0.5, 0.5, 0.5, 0.1, stop=True)  # 0.25 m needed to stop
    assert (late.v_end, late.brake_s) == pytest.approx((math.sqrt(0.15), 0.0))
    assert late.duration_s == pytest.approx((0.5 - math.sqrt(0.15)) / 0.5)


def test_degenerate_inputs():
    assert plan_move(0.0, 0.5, INF, INF, 1.0, stop=True).duration_s == pytest.approx(2.0)
    assert plan_move(0.3, 0.5, 0.5, 0.5, 0.0, stop=True).duration_s == 0.0
    assert plan_move(0.3, 0.5, 0.5, 0.5, 0.0, stop=False).v_end == pytest.approx(0.3)
    assert plan_move(0.9, 0.5, 0.5, 0.5, 1.0, stop=True).duration_s == pytest.approx(2.5)  # v0 clamped to vmax


def test_distance_at_follows_the_phases():
    p = plan_move(0.0, 0.5, 0.5, 0.5, 1.0, stop=True)
    got = [distance_at(p, t) for t in (-1.0, 0.5, 1.5, 2.5, 9.0)]
    assert got == pytest.approx([0.0, 0.0625, 0.5, 0.9375, 1.0])
    assert distance_at(plan_move(0.0, 0.5, INF, INF, 1.0, stop=True), 1.0) == pytest.approx(0.5)
