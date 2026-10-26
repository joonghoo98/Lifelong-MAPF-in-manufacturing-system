import pytest

from agvtb.planning.m6_gate import RULES, check_m6_gate, longest_hop, rest_cells, station_hops, theorem_c
from agvtb.world.cells import build_cell_map, cell_map_from_graph
from agvtb.world.loader import load_world
from helpers import CONFIGS, SMALL_GRAPHS

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
    return request.param, build_cell_map(lw.world, lw.graph, lw.conflicts)


def test_the_gate_of_the_three_maps_at_their_limit(built):
    name, cm = built
    for rule, q in zip(RULES, MAP_REST[name]):
        assert len(rest_cells(cm, rule)) == q and check_m6_gate(cm, rule, q).ok
        refused = check_m6_gate(cm, rule, q + 1)
        assert refused.failed == ("C1",) and not refused.ok and refused.violations[0].startswith("C1: ")
    if name != "songdo_grid_bi":
        assert longest_hop(cm) == 5


def test_the_report_as_data(built):
    name, cm = built
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
