from collections import Counter

import pytest

from agvtb.planning.eviction import EvictionPlanner, Fail
from agvtb.planning.intervals import EPS
from agvtb.planning.occupancy import Occupancy
from agvtb.planning.split_sipp import SplitSipp
from agvtb.validate.plans import pairwise_overlaps
from helpers import cells_of, evict_world, occupancy_of, rest_at


def setup(lw, cm, rest, paths, budget=None):
    _, cons, ends, homes = occupancy_of(lw, cm, rest, paths)
    stats = Counter()
    return EvictionPlanner(lw.graph, lw.conflicts, cm, rest, stats, budget=budget), stats, cons, ends, homes


def clean(lw, b, ends, before=None):
    """No two vehicles' whole paths collide; before[v] is v's path up to its end element."""
    before = before or {}
    full = {v: before.get(v, [])[:-1] + b.paths.get(v, [ends[v]]) for v in ends}
    return pairwise_overlaps(full, lw.conflicts) == []


def chain(tmp_path):
    lw, cm = evict_world(tmp_path, "chain")
    rest = cells_of(cm, "A", "B", "C", "D", "F", "G", "W")
    at = {0: "A", 1: "B", 2: "C", 3: "D"}
    return (lw, cm, rest) + setup(lw, cm, rest, {v: rest_at(lw, n) for v, n in at.items()})


def test_a_dead_end_is_left_before_r_goes_in(tmp_path):  # CE-2 and CE-3, spec test 2 (spec line 257)
    lw, cm = evict_world(tmp_path, "deadend")
    rest = cells_of(cm, "v0", "v1", "v3", "o3")
    planner, stats, cons, ends, homes = setup(lw, cm, rest, {1: rest_at(lw, "v3"), 2: rest_at(lw, "v1")})
    b = planner.plan(2, ["v4", "o2"], ends, cons, homes, 0.0)
    assert b is not None and b.fast and set(b.order) == {1, 2} and b.order[-1] == 2
    assert b.homes[1] != cm.cell_of["v3"] and clean(lw, b, ends)


def test_a_chain_of_three_goes_out_outer_first(tmp_path):  # CE-5, spec test 3 (spec line 258); decision 2
    lw, cm, rest, planner, stats, cons, ends, homes = chain(tmp_path)
    b = planner.plan(0, ["E", "U"], ends, cons, homes, 0.0)
    assert b is not None and b.order == (1, 2, 3, 0)
    assert sorted(b.homes[v] for v in (1, 2, 3)) == list(cells_of(cm, "F", "G", "W")) and clean(lw, b, ends)
    assert stats["m6_chain_max"] == 2 and stats["m6_fast_evicted"] == 3


def test_an_evictee_may_rest_in_the_start_cell_r_left(tmp_path):  # Review Focus 3
    lw, cm = evict_world(tmp_path, "loop")
    planner, _, cons, ends, homes = setup(lw, cm, cells_of(cm, "A", "B"), {1: rest_at(lw, "A"), 2: rest_at(lw, "B")})
    b = planner.plan(1, ["P", "U"], ends, cons, homes, 0.0)
    assert b is not None and b.homes == {2: cm.cell_of["A"], 1: cm.cell_of["B"]} and clean(lw, b, ends)


def test_an_evictee_on_its_way_leaves_after_it_arrives(tmp_path):  # Review Focus 1
    lw, cm = evict_world(tmp_path, "loop")
    g, mv = lw.graph, "M,T,B"
    s_t = next(s for s in sorted(g) if g[s].kind == "S" and mv in g[s].next)
    t_arr = g[mv].duration_s
    way = [(s_t, 0.0), (mv, 0.0), (g[mv].next[0], t_arr)]
    planner, _, cons, ends, homes = setup(lw, cm, cells_of(cm, "A", "B", "T"), {1: rest_at(lw, "A", 10.0), 2: way})
    b = planner.plan(1, ["P", "U"], ends, cons, homes, 0.0)
    assert b is not None and 2 in b.order
    assert b.paths[2][0] == way[-1] and b.paths[2][1][1] >= t_arr - EPS and clean(lw, b, ends, {2: way})


def test_tail_evictions_see_r_path(tmp_path):  # decision 1, the gap of spec lines 134-153
    lw, cm = evict_world(tmp_path, "tail")
    planner, stats, cons, ends, homes = setup(lw, cm, cells_of(cm, "R0", "D"),
                                             {1: rest_at(lw, "R0"), 2: rest_at(lw, "D")})
    b = planner.plan(1, ["L1", "U"], ends, cons, homes, 0.0)
    assert b is None or clean(lw, b, ends)
    assert b is not None or stats["m6_fail_tail_evict"] + stats["m6_fail_tail_hard"] == 1


def test_a_failed_plan_leaves_the_occupancy_as_it_was(tmp_path, monkeypatch):  # spec CE-5 (1): a transaction
    lw, cm, rest, planner, _, cons, ends, homes = chain(tmp_path)
    db = Occupancy.from_sim(lw.graph, lw.conflicts, cm, rest, 0.0, cons, ends, homes)
    snap = (repr(db._entries), dict(db.rests), dict(db.homes))
    monkeypatch.setattr(EvictionPlanner, "_evict", lambda self, *a, **k: False)
    with pytest.raises(Fail) as err:
        planner._verified(db, 0, ends[0][0], ends[0][1], ["E", "U"], False, True)
    assert err.value.branch == "evict" and (repr(db._entries), dict(db.rests), dict(db.homes)) == snap


def test_a_failed_eviction_tries_stepping_aside(tmp_path, monkeypatch):  # algorithm 3
    lw, cm, rest, planner, stats, cons, ends, homes = chain(tmp_path)
    monkeypatch.setattr(EvictionPlanner, "_evict", lambda self, *a, **k: False)
    assert planner.plan(0, ["E", "U"], ends, cons, homes, 0.0) is None
    assert stats["m6_sidestep_try"] == 1 and stats["m6_fail_self_clear"] == 1 and stats["m6_fast_fail"] == 1


@pytest.mark.parametrize("may_recall,recalls,branch", [(False, 0, "hard"), (True, 1, None)])
def test_the_recall_happens_at_most_once(tmp_path, monkeypatch, may_recall, recalls, branch):  # U2 ③
    lw, cm, rest, planner, stats, cons, ends, homes = chain(tmp_path)
    real = SplitSipp.staged
    monkeypatch.setattr(SplitSipp, "staged",
                        lambda self, occ, me, start, t0, legs, locked, split:
                        real(self, occ, me, start, t0, legs, locked, split) if split else None)
    db = Occupancy.from_sim(lw.graph, lw.conflicts, cm, rest, 0.0, cons, ends, homes)
    if branch is None:
        planner._verified(db, 0, ends[0][0], ends[0][1], ["E", "U"], False, may_recall)
    else:
        with pytest.raises(Fail) as err:
            planner._verified(db, 0, ends[0][0], ends[0][1], ["E", "U"], False, may_recall)
        assert err.value.branch == branch
    assert stats["m6_recall"] == recalls


def test_a_probe_over_budget_is_one_fail(tmp_path):
    lw, cm = evict_world(tmp_path, "loop")
    planner, stats, cons, ends, homes = setup(lw, cm, cells_of(cm, "A", "B"),
                                             {1: rest_at(lw, "A"), 2: rest_at(lw, "B")}, budget=1)
    assert planner.plan(1, ["P", "U"], ends, cons, homes, 0.0) is None
    assert stats["m6_fail_probe"] == 1 and stats["m6_fast_fail"] == 1 and stats["m6_search_budget"] >= 1
