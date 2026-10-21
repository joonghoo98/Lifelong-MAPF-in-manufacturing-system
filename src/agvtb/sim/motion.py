"""Trapezoid and triangle speed profiles of one straight move (DAIM agv/agv_motion.py:33-131, same cases).

A move of `length` metres starts at speed v0 (clamped to [0, vmax]) and either stops at its end or passes
through at the speed it reaches. The keyframes are (time, distance, speed) at the phase boundaries.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MovePlan:
    duration_s: float
    v_end: float
    brake_s: float  # when braking starts (the pass-or-stop decision point); duration_s if it never brakes
    keyframes: tuple[tuple[float, float, float], ...]  # (t, distance, speed)


def plan_move(v0: float, vmax: float, accel: float, decel: float, length: float, stop: bool) -> MovePlan:
    """The fastest profile under vmax, accel and decel: stop at the end if stop, else pass through."""
    v0 = max(0.0, min(v0, vmax))
    if length <= 1e-9:
        return MovePlan(0.0, 0.0 if stop else v0, 0.0, ((0.0, 0.0, v0),))
    if not (math.isfinite(accel) and math.isfinite(decel)):
        t = length / vmax
        v_end = 0.0 if stop else vmax
        return MovePlan(t, v_end, t, ((0.0, 0.0, vmax), (t, length, v_end)))
    t_acc = (vmax - v0) / accel
    d_acc = (v0 + vmax) * 0.5 * t_acc
    if not stop:
        if d_acc >= length:
            v_end = math.sqrt(v0 * v0 + 2.0 * accel * length)
            t_end = (v_end - v0) / accel
            return MovePlan(t_end, v_end, t_end, ((0.0, 0.0, v0), (t_end, length, v_end)))
        t = t_acc + (length - d_acc) / vmax
        return MovePlan(t, vmax, t, ((0.0, 0.0, v0), (t_acc, d_acc, vmax), (t, length, vmax)))
    if v0 * v0 / (2.0 * decel) > length + 1e-9:  # cannot stop within the move
        v_end = math.sqrt(max(0.0, v0 * v0 - 2.0 * decel * length))
        t_end = (v0 - v_end) / decel
        return MovePlan(t_end, v_end, 0.0, ((0.0, 0.0, v0), (t_end, length, v_end)))
    d_dec = vmax * vmax / (2.0 * decel)
    if d_acc + d_dec <= length + 1e-9:  # trapezoid
        d_cruise = length - d_acc - d_dec
        brake = t_acc + d_cruise / vmax
        t = brake + vmax / decel
        return MovePlan(t, 0.0, brake, ((0.0, 0.0, v0), (t_acc, d_acc, vmax), (brake, d_acc + d_cruise, vmax),
                                        (t, length, 0.0)))
    vp2 = (length + v0 * v0 / (2.0 * accel)) / (1.0 / (2.0 * accel) + 1.0 / (2.0 * decel))  # triangle
    vp = min(math.sqrt(max(0.0, vp2)), vmax)
    t_acc2 = (vp - v0) / accel
    d_acc2 = (v0 + vp) * 0.5 * t_acc2
    t = t_acc2 + vp / decel
    return MovePlan(t, 0.0, t_acc2, ((0.0, 0.0, v0), (t_acc2, d_acc2, vp), (t, length, 0.0)))


def distance_at(plan: MovePlan, t: float) -> float:
    """Distance covered at time t since the move started; each phase has a constant acceleration."""
    frames = plan.keyframes
    if t <= 0.0:
        return 0.0
    if t >= plan.duration_s:
        return frames[-1][1]
    for (t0, d0, v0), (t1, d1, _v1) in zip(frames, frames[1:]):
        if t <= t1:
            span = t1 - t0
            if span <= 0.0:
                return d1
            a = 2.0 * (d1 - d0 - v0 * span) / (span * span)
            dt = t - t0
            return d0 + v0 * dt + 0.5 * a * dt * dt
    return frames[-1][1]
