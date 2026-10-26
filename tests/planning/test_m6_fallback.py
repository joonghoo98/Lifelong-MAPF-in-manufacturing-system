import random

import pytest

import agvtb.planning.m6_fallback as m6_fallback
from agvtb.planning.intervals import EPS
from agvtb.planning.m6_fallback import M6Fallback
from agvtb.planning.m6_gate import rest_cells
from agvtb.planning.offline import initial_stop
from agvtb.planning.reservations import path_constraints
from agvtb.validate.plans import pairwise_overlaps, task_order_violation
from agvtb.world.cells import build_cell_map
from agvtb.world.loader import load_world
from helpers import CONFIGS


@pytest.fixture(scope="module")
def kairos():
    lw = load_world(CONFIGS / "KAIROS_AMR.json")
    return lw, build_cell_map(lw.world, lw.graph, lw.conflicts)


def resting(lw, cells):
    """Every vehicle on the first stop of its cell from time 0, reserved forever."""
    ends = {v: (initial_stop(lw.graph, c), 0.0) for v, c in enumerate(cells)}
    return ends, {v: path_constraints([e], lw.graph, lw.conflicts) for v, e in ends.items()}


def check_bundle(lw, cm, rest, b, legs, ends):  # Review Focus 4: paths start at path[-1], no move before t0
    rest_nodes = {n for c in rest for n in cm.members[c]}
    assert task_order_violation(b.paths[b.r], legs, rest_nodes) is None
    assert b.order[-1] == b.r and set(b.order) == set(b.paths) == set(b.homes) and b.t0 >= b.t_star
    for u, p in b.paths.items():
        assert p[0] == ends[u] and p[1][1] >= b.t0 - EPS and p[-1][0].startswith("S,")
        assert cm.cell_of[lw.graph[p[-1][0]].nodes[0]] == b.homes[u]
    cells = [b.homes.get(u, cm.cell_of[lw.graph[e[0]].nodes[0]]) for u, e in sorted(ends.items())]
    assert len(set(cells)) == len(cells) and set(cells) <= set(rest)
    assert pairwise_overlaps({u: b.paths.get(u, [e]) for u, e in ends.items()}, lw.conflicts) == []
    busy = sorted((p[i][1], p[i + 1][1]) for p in b.paths.values() for i in range(len(p) - 1)
                  if not p[i][0].startswith("S,"))
    assert all(a[1] <= c[0] + EPS for a, c in zip(busy, busy[1:]))  # one vehicle at a time


def test_a_job_on_a_full_kairos(kairos):  # branch: 26 vehicles on the 26 rest cells
    lw, cm = kairos
    rest = rest_cells(cm, "branch")
    ends, cons = resting(lw, rest)
    fb, ports = M6Fallback(lw.graph, lw.conflicts, cm, rest, 26), sorted(lw.world.port_nodes)
    legs = [ports[0], ports[-1]]
    b = fb.plan(0, legs, ends, cons)
    assert b is not None and (b.t_star, b.t0) == (0.0, 0.0)
    check_bundle(lw, cm, rest, b, legs, ends)
    assert fb.stats["m6_fallback_used"] == 1 and fb.stats["m6_cell_moves"] == b.moves > 0


def test_no_member_moves_before_t_star(kairos):
    lw, cm = kairos
    rest = rest_cells(cm, "branch")
    ends, cons = resting(lw, rest)
    cons[5] = cons[5] + [("S,nowhere,0", 0.0, 500.0)]  # another vehicle holds something until 500 s
    b = M6Fallback(lw.graph, lw.conflicts, cm, rest, 26).plan(0, ["na.5", "na.9"], ends, cons)
    assert (b.t_star, b.t0) == (500.0, 500.0) and all(p[1][1] >= 500.0 for p in b.paths.values())


def test_two_ports_in_one_cell(kairos):  # Review Focus 1
    lw, cm = kairos
    rest = rest_cells(cm, "branch")
    ends, cons = resting(lw, rest)
    p = M6Fallback(lw.graph, lw.conflicts, cm, rest, 26).plan(0, ["na.5", "na.9"], ends, cons).paths[0]
    i = next(k for k, (s, _) in enumerate(p) if s.startswith("L,na.5,"))
    j = next(k for k, (s, _) in enumerate(p) if s.startswith("L,na.9,"))
    assert i < j and all(set(lw.graph[s].nodes) <= set(cm.members["na.10"]) for s, _ in p[i:j + 1])


@pytest.mark.parametrize("rule,n", [("branch", 26), ("all", 31)])
def test_kairos_jobs_one_after_another(kairos, rule, n):
    lw, cm = kairos
    rest, rng = rest_cells(cm, rule), random.Random(5)
    ends, cons = resting(lw, rng.sample(list(rest), n))
    fb, ports = M6Fallback(lw.graph, lw.conflicts, cm, rest, n), sorted(lw.world.port_nodes)
    for _ in range(30):
        r, legs = rng.randrange(n), rng.sample(ports, 2)
        b = fb.plan(r, legs, ends, cons)
        assert b is not None
        check_bundle(lw, cm, rest, b, legs, ends)
        for u, p in b.paths.items():
            ends[u], cons[u] = p[-1], path_constraints(p, lw.graph, lw.conflicts)


def test_a_bundle_that_would_collide_is_refused(kairos, monkeypatch):  # the overlap check is the safety net
    lw, cm = kairos
    rest = rest_cells(cm, "branch")
    c, d = next((c, d) for c in rest for d in cm.adj[c] if d in rest and len(cm.members[d]) == 1)  # d: one node
    ends, cons = resting(lw, [c, d] + [x for x in rest if x not in (c, d)])
    monkeypatch.setattr(m6_fallback, "solve", lambda cm_, rest_, hops, at, r, legs: ([("move", 0, c, d)], {**at, 0: d}))
    fb = M6Fallback(lw.graph, lw.conflicts, cm, rest, 26)
    assert fb.plan(0, ["na.5", "na.9"], ends, cons) is None
    assert fb.stats["m6_bundle_overlap"] == fb.stats["m6_fallback_fail"] == 1
