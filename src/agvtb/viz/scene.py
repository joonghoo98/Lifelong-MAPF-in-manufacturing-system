"""What the viewer draws at one time, as plain data (no pygame): vehicles, ports with their waiting jobs, the HUD."""
from dataclasses import dataclass

from agvtb.sim.executor import wait_cycles
from agvtb.sim.pose import pose_of

PHASE_LETTER = {"IDLE": "I", "TO_PICKUP": "P", "LOADING": "L", "TO_DELIVERY": "D", "UNLOADING": "U"}


@dataclass(frozen=True)
class VehicleView:
    vid: int
    x: float
    y: float
    heading_deg: float
    kind: str  # "move", "rotate", "blocked", "dwell" or "rest"
    phase: str  # I, P, L, D or U
    in_cycle: bool  # in a wait cycle now


@dataclass(frozen=True)
class PortView:
    node: str
    x: float
    y: float
    waiting: int


@dataclass(frozen=True)
class Scene:
    t: float
    vehicles: tuple[VehicleView, ...]
    ports: tuple[PortView, ...]
    hud: tuple[str, ...]


def _fmt(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def build_scene(sim, speed: float, paused: bool, warmup: float | None = None) -> Scene:
    """The scene of sim at sim.now. With warmup, two more HUD lines: the KPIs of [warmup, now] once the warm-up is
    over (DAIM's rule, research §F), else the time left in it."""
    t = sim.now
    cycling = {vid for group in wait_cycles(sim.executor.blocked_by()) for vid in group}
    vehicles = []
    for vid in sorted(sim.executor.vehicles):
        p = pose_of(sim.world, sim.graph, sim.executor, vid, t)
        vehicles.append(VehicleView(vid, p.x, p.y, p.heading_deg, p.kind, PHASE_LETTER[sim.mcs.phase[vid]],
                                    vid in cycling))
    waiting: dict[str, int] = {}
    for job in sim.mcs.waiting():
        waiting[job.src] = waiting.get(job.src, 0) + 1
    ports = tuple(PortView(n, *sim.world.nodes[n], waiting.get(n, 0)) for n in sorted(set(sim.world.port_nodes)))
    done = sum(1 for j in sim.mcs.jobs if j.t_completed is not None)
    hud = (f"t {t:9.1f} s  x{speed:g}{'  paused' if paused else ''}",
           f"done {done} waiting {len(sim.mcs.waiting())} cycles {len(sim.wait_cycles)}"
           f"{'  FROZEN' if sim.stopped else ''}")
    if warmup is not None:
        if t < warmup:
            hud += (f"warm-up: {warmup - t:.0f} s left", "")
        else:
            k = sim.kpi(warmup, t)
            hud += (f"KPI {warmup:g}-{t:.0f} s: throughput {_fmt(k['throughput_per_h'], '.1f')}/h  "
                    f"utilization {_fmt(k['utilization'], '.2f')}",
                    f"cycle {_fmt(k['cycle_s'], '.1f')} s  delivery {_fmt(k['deliver_s'], '.1f')} s  done {k['completed']}")
    return Scene(t, tuple(vehicles), ports, hud)
