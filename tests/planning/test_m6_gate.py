from dataclasses import replace

import pytest

from agvtb.planning.intervals import EPS
from agvtb.planning.m6_gate import (RULES, cell_facts, check_m6_gate, longest_hop, map_assumptions, rest_cells,
                                    state_facts, station_hops, theorem_c)
from agvtb.world.cells import build_cell_map, cell_map_from_graph
from agvtb.world.loader import load_world
from helpers import CONFIGS, SMALL_GRAPHS, evict_world, one_way_tiny_config, split_config, world_of_config

REFERENCE = [  # (graph, rule, N, failed): R26 check_c.theorem_c (ref_cases.out.txt), its T1..T4 being C1..C4
    ("lollipop", "branch", 3, ()), ("lollipop", "branch", 4, ("C2",)), ("lollipop", "branch", 5, ("C1", "C2")),
    ("lollipop", "all", 5, ("C2", "C3")), ("cutport", "no-cut", 4, ("C1", "C2")), ("cutport", "all", 5, ("C2", "C3")),
    ("spider", "branch", 3, ()), ("spider", "branch", 4, ("C3",)), ("spider", "branch", 5, ("C2", "C3")),
    ("comb", "all", 5, ()), ("comb", "all", 6, ("C2",)),
    ("longlane", "branch", 7, ("C1", "C2", "C3", "C4")), ("longlane", "all", 7, ("C2", "C3", "C4")),
]
REST = {  # R26 rest_nodes on the same graphs
    "lollipop": (("1", "3"), ("1", "3", "4", "5"), ("0", "1", "3", "4", "5")),
    "spider": (("7",), ("0", "1", "2", "4", "5", "7"), ("0", "1", "2", "4", "5", "7")),
    "longlane": (("2", "8"), ("2", "3", "4", "6", "7", "8"), ("0", "2", "3", "4", "6", "7", "8")),
}
MAP_REST = {"KAIROS_AMR": (13, 26, 31), "mini_deadend": (146, 146, 150), "songdo_grid_bi": (1363, 1364, 1374)}


@pytest.mark.parametrize("name,rule,n,failed", REFERENCE)
def test_theorem_c_gives_the_reference_verdicts(name, rule, n, failed):
    cm = cell_map_from_graph(*SMALL_GRAPHS[name])
    assert theorem_c(cm, rest_cells(cm, rule), n) == failed


def test_rest_cells_of_the_small_graphs():
    for name, want in REST.items():
        cm = cell_map_from_graph(*SMALL_GRAPHS[name])
        assert tuple(rest_cells(cm, rule) for rule in RULES) == want


def test_station_hops_of_the_lollipop():
    cm = cell_map_from_graph(*SMALL_GRAPHS["lollipop"])
    assert station_hops(cm, 4)["1"] == {"2": ("2",), "3": ("0", "3"), "6": ("0", "4", "5", "6")}
    assert station_hops(cm, 3)["1"] == {"2": ("2",), "3": ("0", "3")} and station_hops(cm, 3)["6"] == {}
    assert longest_hop(cm) == 4


@pytest.fixture(scope="module", params=sorted(MAP_REST))
def built(request):
    lw = load_world(CONFIGS / f"{request.param}.json")
    return request.param, lw, build_cell_map(lw.world, lw.graph, lw.conflicts)


def test_the_gate_of_the_three_maps_at_their_limit(built):
    name, _, cm = built
    for rule, q in zip(RULES, MAP_REST[name]):
        assert len(rest_cells(cm, rule)) == q and check_m6_gate(cm, rule, q).ok
        refused = check_m6_gate(cm, rule, q + 1)
        assert refused.failed == ("C1",) and not refused.ok and refused.violations[0].startswith("C1: ")
    if name != "songdo_grid_bi":
        assert longest_hop(cm) == 5


def test_the_report_as_data(built):
    name, _, cm = built
    if name == "KAIROS_AMR":
        assert check_m6_gate(cm, "branch", 26).summary() == {
            "ok": True, "violations": [], "rule": "branch", "vehicles": 26, "cells": 44, "rest": 26,
            "longest_hop": 5, "failed": []}


def test_bad_inputs():
    cm = cell_map_from_graph(*SMALL_GRAPHS["lollipop"])
    with pytest.raises(ValueError):
        check_m6_gate(cm, "branch", 0)
    with pytest.raises(ValueError):
        rest_cells(cm, "loop")


def test_the_map_assumptions(tmp_path):  # final review I1, M1: the theorems want two-way segments and one map
    assert all(map_assumptions(load_world(CONFIGS / f"{name}.json").world) == () for name in sorted(MAP_REST))
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    one_way = map_assumptions(world_of_config(one_way_tiny_config(tmp_path / "a")))
    split = map_assumptions(world_of_config(split_config(tmp_path / "b")))
    assert len(one_way) == 1 and one_way[0].startswith("map: ") and "('B', 'C')" in one_way[0]
    assert len(split) == 1 and split[0].startswith("map: ") and "not connected" in split[0]


def test_the_cell_facts_hold_on_the_three_maps(built):  # R32 I-2: Theorem B's unstated facts, on the testbed cells
    _, lw, cm = built
    assert cell_facts(cm, lw.graph, lw.conflicts, lw.world.port_nodes) == ()


def without_steps_into(graph, drop):
    """graph with every step into a state that drop picks taken out."""
    return {sid: replace(st, next=tuple(n for n in st.next if not drop(graph[n]))) for sid, st in graph.items()}


def test_cell_facts_find_stops_that_cannot_leave_their_cell(tmp_path):  # R32 I-2 (ii): the move inside C u C'
    lw, cm = evict_world(tmp_path, "loop")
    graph = without_steps_into(lw.graph, lambda st: st.kind == "M" and st.nodes[0] == "A")
    found = cell_facts(cm, graph, lw.conflicts, lw.world.port_nodes)
    assert len(found) == 1 and found[0].startswith("cells: ") and "neighbour" in found[0] and "S,A," in found[0]


def test_cell_facts_find_a_port_whose_load_is_out_of_reach(tmp_path):  # R32 I-2 (iii): the port's L inside its cell
    lw, cm = evict_world(tmp_path, "loop")
    graph = without_steps_into(lw.graph, lambda st: st.kind == "L" and st.nodes[0] == "P")
    found = cell_facts(cm, graph, lw.conflicts, lw.world.port_nodes)
    assert len(found) == 1 and found[0].startswith("cells: ") and "L state" in found[0] and "'P'" in found[0]


def test_cell_facts_find_a_cell_without_a_stop(tmp_path):  # R32 I-2: a cell must hold a stop to rest or wait on
    lw, cm = evict_world(tmp_path, "loop")
    graph = {sid: replace(st, next=tuple(n for n in st.next if not n.startswith("S,U,")))
             for sid, st in lw.graph.items() if not sid.startswith("S,U,")}
    found = cell_facts(cm, graph, lw.conflicts, lw.world.port_nodes)
    assert any(f.startswith("cells: ") and "no stop" in f and "'U'" in f for f in found)


def test_cell_facts_find_conflicts_that_are_not_symmetric(tmp_path):  # R32 I-2: Theorem B's property 1 needs it
    lw, cm = evict_world(tmp_path, "loop")
    a = next(s for s in sorted(lw.graph) if s.startswith("S,A,"))
    b = next(s for s in sorted(lw.graph) if s.startswith("S,U,"))
    assert b not in lw.conflicts[a]
    found = cell_facts(cm, lw.graph, {**lw.conflicts, a: lw.conflicts[a] | {b}}, lw.world.port_nodes)
    assert len(found) == 1 and found[0].startswith("cells: ") and "not symmetric" in found[0] and a in found[0]


def test_the_state_facts_hold_on_the_three_maps(built):  # R34 D9 · D12, user decision 2026-10-28
    _, lw, _ = built
    assert state_facts(lw.graph, lw.conflicts) == ()


@pytest.mark.parametrize("kind,length", [("L", 0.0), ("M", 1e-12), ("R", EPS)])
def test_state_facts_find_an_action_that_lasts_eps_or_less(tmp_path, kind, length):  # R34 D9; the config takes dwell 0
    lw, _ = evict_world(tmp_path, "loop")
    first = next(sid for sid in sorted(lw.graph) if lw.graph[sid].kind == kind)
    found = state_facts({**lw.graph, first: replace(lw.graph[first], duration_s=length)}, lw.conflicts)
    assert len(found) == 1 and found[0].startswith("states: ") and "D9" in found[0] and first in found[0]


def test_state_facts_find_a_move_that_misses_what_its_stop_hits(tmp_path):  # R34 D12 (P)
    lw, _ = evict_world(tmp_path, "loop")
    m, s, x = "M,A,B", "S,B,0", "M,B,P"  # x takes node B, so on the grid rule it hits the stop s and the move m into s
    assert s in lw.graph[m].next and x in lw.conflicts[s] and x in lw.conflicts[m]
    assert state_facts(lw.graph, lw.conflicts) == ()
    cut = {**lw.conflicts, m: lw.conflicts[m] - {x}, x: lw.conflicts[x] - {m}}
    found = state_facts(lw.graph, cut)
    assert len(found) == 1 and found[0].startswith("states: ") and "(P)" in found[0]
    assert m in found[0] and s in found[0] and x in found[0] and found[0].startswith("states: 1 ")
