"""Vehicle execution on the TAPG: claims, moves with pass-through, rotations, dwells, wake-ups (DAIM agv_des_env.py).

The rules are DAIM's (research §B3, §B4, §B7, §B9, §C2):
- TRY_ADVANCE walks over stop states; a vehicle leaves a stop only for a next M/R/L node no other vehicle's node
  precedes, and for an M/R only if it can claim the whole unit "M/R + the stops after it" (env:555-676, 909-942).
- A move is committed as a stopping profile; at its braking point the vehicle passes through if the next non-stop
  state is a move it can claim, else it stops at the end (env:1095-1160, 1187-1212). After a move it chains into
  the next move without stopping when it can (env:1247-1289).
- At an arrival every node of the vehicle before its new position leaves the graph and the vehicles waiting on
  them are woken just after (t + 1e-9) (env:1057-1083, 808-838). A load leaves at LOAD_DONE (env:520-553).
Every event has rank AGV. on_advance(vid, path_idx, t) is called after each M/R arrival and each LOAD_DONE.
"""
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from agvtb.sim.kernel import AGV, Kernel
from agvtb.sim.motion import MovePlan, plan_move
from agvtb.sim.tapg import Key, Tapg
from agvtb.world.model import World
from agvtb.world.states import StateGraph

Geometry = Callable[[str], tuple[float, float]]
WAKE_DELAY = 1e-9


def world_geometry(world: World) -> Geometry:
    """M state id -> (segment length m, top speed m/s)."""
    def geometry(sid: str) -> tuple[float, float]:
        _, u, v = sid.split(",")
        seg = world.segments[(u, v)]
        return seg.length_m, seg.speed_mps
    return geometry


@dataclass
class Vehicle:
    vid: int
    state: str = "IDLE"  # IDLE, MOVING, ROTATING, WAITING (blocked or dwelling) or DONE
    path_idx: int = 0
    claim_idx: int = 0
    v: float = 0.0
    dwelling: bool = False
    executed: list[tuple[str, float]] = field(default_factory=list)  # each state entered, and when
    node: Key | None = None  # the M/R in progress
    t0: float = 0.0  # when the move or rotation started
    dur: float = 0.0  # its duration as now committed
    move: tuple[float, float, float] = (0.0, 0.0, 0.0)  # v0, vmax, length of the move
    v_end: float = 0.0
    profile: MovePlan | None = None  # the move's profile as now committed (stop, then the braking decision's)
    load_tok: int = 0
    load_end: float | None = None
    load_sid: str | None = None


class Executor:
    def __init__(self, graph: StateGraph, tapg: Tapg, kernel: Kernel, geometry: Geometry, accel: float,
                 decel: float, on_advance: Callable[[int, int, float], None],
                 on_event: Callable[[float, str, int, tuple], None] = lambda t, kind, vid, data: None):
        self.graph = graph
        self.tapg = tapg
        self.kernel = kernel
        self.geometry = geometry
        self.accel = accel
        self.decel = decel
        self.on_advance = on_advance
        self.on_event = on_event
        self.vehicles: dict[int, Vehicle] = {}

    # --- public ------------------------------------------------------------------------------------------------
    def add(self, vid: int) -> None:
        sid, t = self.tapg.paths[vid][0]
        start = max(self.kernel.now, t)
        self.vehicles[vid] = Vehicle(vid, executed=[(sid, start)])
        self.kernel.schedule(start, AGV, self._try_advance, vid)

    def kick(self, vid: int, t: float) -> None:
        """A path was appended: a vehicle that had finished starts again from its last state."""
        v = self.vehicles[vid]
        if v.state == "DONE":
            v.state, v.v = "IDLE", 0.0
            v.path_idx = max(0, v.path_idx - 1)
            self.kernel.schedule(t, AGV, self._try_advance, vid)

    def revocable(self, vid: int, k: int) -> bool:
        """The states from index k on are neither entered nor claimed, so they may be taken back (env:1561-1565)."""
        v = self.vehicles[vid]
        return k > v.path_idx and k >= v.claim_idx

    def shift(self, vid: int, n: int) -> None:
        """After tapg.prune(vid, n): the indices move back by n."""
        v = self.vehicles[vid]
        v.path_idx -= n
        v.claim_idx = max(0, v.claim_idx - n)

    def wake_ready(self, t: float) -> None:
        """After a recompute: wake the blocked vehicles that can now go on (DAIM env:1950-1977)."""
        for vid in sorted(self.vehicles):
            v = self.vehicles[vid]
            path = self.tapg.paths[vid]
            if v.state != "WAITING" or v.dwelling or v.path_idx >= len(path):
                continue
            i, go = v.path_idx, True
            while i < len(path):
                if not self._claimable(v, i):
                    go = False
                    break
                if not path[i][0].startswith("S,"):
                    break
                i += 1
            if go:
                v.state = "IDLE"
                self.on_event(t, "wake", vid, ())
                self.kernel.schedule(t + WAKE_DELAY, AGV, self._try_advance, vid)

    def inprog_end(self) -> dict[Key, float]:
        """The real end of every move, rotation and dwell in progress, keyed by its node."""
        out: dict[Key, float] = {}
        for v in self.vehicles.values():
            if v.state in ("MOVING", "ROTATING"):  # keyed by its node now: recompute re-keys nodes (env:1947)
                out[self.tapg.key(v.vid, v.path_idx)] = v.t0 + v.dur
            elif v.dwelling and v.load_end is not None and v.path_idx < len(self.tapg.paths[v.vid]):
                out[self.tapg.key(v.vid, v.path_idx)] = v.load_end
        return out

    def blocked_by(self) -> dict[int, set[int]]:
        """Each vehicle that is not moving -> the other vehicles whose nodes hold its next claim unit."""
        out: dict[int, set[int]] = {}
        for vid in sorted(self.vehicles):
            v = self.vehicles[vid]
            path = self.tapg.paths[vid]
            if v.state not in ("IDLE", "WAITING") or v.path_idx >= len(path):
                continue
            i, seen_action = v.path_idx, False
            while i < len(path):
                sid = path[i][0]
                if seen_action and not sid.startswith("S,"):
                    break
                key = self.tapg.key(vid, i)
                owners = {p[1] for p in self.tapg.preds(key) if p[1] != vid}
                if owners:
                    out[vid] = owners
                    break
                seen_action = seen_action or sid.startswith(("M,", "R,"))
                i += 1
        return out

    # --- helpers -----------------------------------------------------------------------------------------------
    def _enter(self, v: Vehicle, t: float) -> None:
        path = self.tapg.paths[v.vid]
        if v.path_idx < len(path):
            v.executed.append((path[v.path_idx][0], t))

    def _claimable(self, v: Vehicle, i: int) -> bool:
        return self.tapg.claimable(self.tapg.key(v.vid, i), v.vid)

    def _try_claim_next(self, v: Vehicle) -> bool:
        path = self.tapg.paths[v.vid]
        i = v.claim_idx
        while i < len(path):
            if path[i][0].startswith(("M,", "R,")):
                end = i + 1
                while end < len(path) and path[end][0].startswith("S,"):
                    end += 1
                if not all(self._claimable(v, k) for k in range(i, end)):
                    return False
                v.claim_idx = end
                return True
            i += 1
        return False

    def _block(self, v: Vehicle, t: float) -> None:
        if v.state != "WAITING":
            self.on_event(t, "block", v.vid, (self.tapg.paths[v.vid][v.path_idx][0],))
        v.state, v.v = "WAITING", 0.0

    def wake(self, vids: list[int], t: float) -> None:
        for w in vids:
            other = self.vehicles.get(w)
            if other is not None and other.state == "WAITING":
                other.state = "IDLE"
                self.on_event(t, "wake", w, ())
                self.kernel.schedule(t + WAKE_DELAY, AGV, self._try_advance, w)

    def _remove(self, key: Key, t: float) -> None:
        self.wake(self.tapg.remove(key), t)

    # --- events ------------------------------------------------------------------------------------------------
    def _try_advance(self, t: float, vid: int) -> None:
        v = self.vehicles[vid]
        if v.state in ("MOVING", "ROTATING", "DONE"):
            return
        path = self.tapg.paths[vid]
        while v.path_idx < len(path) and path[v.path_idx][0].startswith("S,"):
            nxt = v.path_idx + 1
            if nxt < len(path):
                next_sid = path[nxt][0]
                if not next_sid.startswith("S,") and nxt >= v.claim_idx:
                    if not self._claimable(v, nxt):
                        break
                    if not next_sid.startswith("L,") and not self._try_claim_next(v):
                        break
            v.path_idx += 1
            self._enter(v, t)
        if v.path_idx >= len(path):
            v.state, v.v = "DONE", 0.0
            return
        sid = path[v.path_idx][0]
        if sid.startswith("L,"):
            v.state, v.v, v.dwelling = "WAITING", 0.0, True
            if v.load_sid == sid and v.load_end is not None and v.load_end >= t - 1e-9:
                return
            v.load_tok += 1
            v.load_sid, v.load_end = sid, t + self.graph[sid].duration_s
            self.on_event(t, "dwell", vid, (sid,))
            self.kernel.schedule(v.load_end, AGV, self._load_done, vid, v.load_tok)
            return
        if sid.startswith("S,"):
            self._block(v, t)
            return
        if v.path_idx >= v.claim_idx:
            if not self._claimable(v, v.path_idx) or not self._try_claim_next(v):
                self._block(v, t)
                return
        if sid.startswith("M,"):
            self._start_move(v, t, None)
        else:
            self._start_rotate(v, t)

    def _start_move(self, v: Vehicle, t: float, v_init: float | None) -> None:
        sid = self.tapg.paths[v.vid][v.path_idx][0]
        length, vmax = self.geometry(sid)
        if v_init is not None:
            v.v = min(v_init, vmax)
        else:
            v.v = 0.0 if math.isfinite(self.accel) else vmax
        v.node, v.state, v.dwelling = self.tapg.key(v.vid, v.path_idx), "MOVING", False
        v.move, v.t0 = (v.v, vmax, length), t
        prof = plan_move(v.v, vmax, self.accel, self.decel, length, stop=True)
        v.dur, v.v_end, v.profile = prof.duration_s, prof.v_end, prof
        self.on_event(t, "move", v.vid, (sid,))
        if prof.brake_s <= 1e-9:
            self._brake_decision(v)
        else:
            self.kernel.schedule(t + prof.brake_s, AGV, self._brake_check, v.vid, t)

    def _brake_check(self, t: float, vid: int, t0: float) -> None:
        v = self.vehicles[vid]
        if v.state == "MOVING" and abs(t0 - v.t0) <= 1e-9:
            self._brake_decision(v)

    def _brake_decision(self, v: Vehicle) -> None:
        through = self._next_move_claimable(v)
        v0, vmax, length = v.move
        prof = plan_move(v0, vmax, self.accel, self.decel, length, stop=not through)
        v.dur, v.v_end, v.profile = prof.duration_s, prof.v_end, prof
        self.kernel.schedule(v.t0 + v.dur, AGV, self._arrive, v.vid, v.t0)

    def _next_move_claimable(self, v: Vehicle) -> bool:
        path = self.tapg.paths[v.vid]
        i = v.path_idx + 1
        while i < len(path) and path[i][0].startswith("S,"):
            i += 1
        if i >= len(path) or not path[i][0].startswith("M,"):
            return False
        return i < v.claim_idx or self._try_claim_next(v)

    def _start_rotate(self, v: Vehicle, t: float) -> None:
        sid = self.tapg.paths[v.vid][v.path_idx][0]
        v.node, v.state, v.dwelling, v.v = self.tapg.key(v.vid, v.path_idx), "ROTATING", False, 0.0
        v.t0, v.dur = t, self.graph[sid].duration_s or 0.0
        self.on_event(t, "rotate", v.vid, (sid,))
        self.kernel.schedule(t + v.dur, AGV, self._arrive, v.vid, t)

    def _arrive(self, t: float, vid: int, t0: float) -> None:
        v = self.vehicles[vid]
        if v.state not in ("MOVING", "ROTATING") or abs(t0 - v.t0) > 1e-9:
            return
        was_move = v.state == "MOVING"
        carry = v.v_end if was_move else 0.0
        v.node = None
        v.path_idx += 1
        v.state, v.v = "IDLE", carry
        self._enter(v, t)
        self.on_event(t, "arrive", vid, (self.tapg.paths[vid][v.path_idx - 1][0],))
        self.on_advance(vid, v.path_idx, t)
        for i in range(v.path_idx):
            self._remove(self.tapg.key(vid, i), t)
        if math.isfinite(self.accel) and carry > 0 and was_move and self._chain(v, carry, t):
            return
        v.v = 0.0
        self.kernel.schedule(t, AGV, self._try_advance, vid)

    def _chain(self, v: Vehicle, carry: float, t: float) -> bool:
        path = self.tapg.paths[v.vid]
        while v.path_idx < len(path) and path[v.path_idx][0].startswith("S,"):
            nxt = v.path_idx + 1
            if nxt < len(path) and not path[nxt][0].startswith("S,") and not self._claimable(v, nxt):
                return False
            v.path_idx += 1
            self._enter(v, t)
        if v.path_idx >= len(path):
            v.state, v.v = "DONE", 0.0
            return True
        if path[v.path_idx][0].startswith("M,") and self._claimable(v, v.path_idx):
            self._start_move(v, t, carry)
            return True
        return False

    def _load_done(self, t: float, vid: int, tok: int) -> None:
        v = self.vehicles[vid]
        if v.state == "DONE" or tok != v.load_tok:
            return
        v.load_end, v.load_sid, v.dwelling = None, None, False
        path = self.tapg.paths[vid]
        if v.path_idx >= len(path):
            return
        if not path[v.path_idx][0].startswith("L,"):
            self.kernel.schedule(t, AGV, self._try_advance, vid)
            return
        self._remove(self.tapg.key(vid, v.path_idx), t)
        v.path_idx += 1
        v.state = "IDLE"
        self._enter(v, t)
        self.on_event(t, "load_done", vid, ())
        self.on_advance(vid, v.path_idx, t)
        self.kernel.schedule(t, AGV, self._try_advance, vid)


def wait_cycles(blocked: Mapping[int, set[int]]) -> list[tuple[int, ...]]:
    """The groups of vehicles (two or more) that wait on each other through some chain of blockers: the strongly
    connected components of the wait-for relation, each as its sorted ids, the list sorted. A vehicle needs every
    blocker to move on, so such a group can never all move."""
    reach: dict[int, set[int]] = {}
    for start in blocked:
        seen, stack = set(), [start]
        while stack:
            for nxt in blocked.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        reach[start] = seen
    groups = {tuple(sorted({v} | {u for u in reach[v] if u in reach and v in reach[u]}))
              for v in blocked if v in reach[v]}
    return sorted(g for g in groups if len(g) >= 2)
