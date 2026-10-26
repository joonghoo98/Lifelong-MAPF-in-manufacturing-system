import itertools
import random

import pytest

from agvtb.planning.cell_procedure import Stuck, solve
from agvtb.planning.m6_gate import RULES, rest_cells, station_hops, theorem_c
from agvtb.world.cells import build_cell_map, cell_map_from_graph
from agvtb.world.loader import load_world
from helpers import CONFIGS, SMALL_GRAPHS


def replay(cm, rest, at, r, legs, steps, final):
    """Every move goes to a free neighbour cell, r loads in each leg's cell in order, all end on distinct rest cells."""
    pos, occ, loads = dict(at), {c: v for v, c in at.items()}, []
    for step in steps:
        if step[0] == "move":
            _, vid, src, dst = step
            assert pos[vid] == src and dst in cm.adj[src] and dst not in occ, step
            del occ[src]
            occ[dst], pos[vid] = vid, dst
        else:
            _, vid, port = step
            assert vid == r and pos[r] == cm.cell_of[port], step
            loads.append(port)
    assert loads == list(legs) and pos == final
    assert len(set(final.values())) == len(final) and set(final.values()) <= set(rest)


def legs_of(ports):
    return [list(p) for p in itertools.permutations(sorted(ports), 2)] or [[p] for p in ports]


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("name", sorted(SMALL_GRAPHS))
def test_every_start_is_solved_where_theorem_c_holds(name, rule):  # Review Focus 2: cutport, spider
    adj, ports = SMALL_GRAPHS[name]
    cm = cell_map_from_graph(adj, ports)
    rest, tried = rest_cells(cm, rule), 0
    for n in range(1, len(rest) + 1):
        if theorem_c(cm, rest, n):
            continue
        hops = station_hops(cm, len(cm.cells) - n)
        for start in rest:
            for others in itertools.combinations([c for c in rest if c != start], n - 1):
                at = {0: start} | {i + 1: c for i, c in enumerate(others)}
                for legs in legs_of(ports):
                    steps, final = solve(cm, rest, hops, at, 0, legs)
                    replay(cm, rest, at, 0, legs, steps, final)
                    tried += 1
    assert tried > 0


def test_without_room_for_the_long_hop_it_is_stuck():  # lollipop branch N = 4: C2 fails, the hop to 6 has 4 cells
    cm = cell_map_from_graph(*SMALL_GRAPHS["lollipop"])
    with pytest.raises(Stuck):
        solve(cm, ("1", "3", "4", "5"), station_hops(cm, 3), {0: "1", 1: "3", 2: "4", 3: "5"}, 0, ["2", "6"])


@pytest.fixture(scope="module")
def kairos():
    lw = load_world(CONFIGS / "KAIROS_AMR.json")
    return lw, build_cell_map(lw.world, lw.graph, lw.conflicts)


@pytest.mark.parametrize("rule,n", [("no-cut", 13), ("branch", 20), ("branch", 26), ("all", 31)])
def test_kairos_jobs_one_after_another(kairos, rule, n):
    lw, cm = kairos
    rest, ports = rest_cells(cm, rule), sorted(lw.world.port_nodes)
    hops, rng = station_hops(cm, len(cm.cells) - n), random.Random(11)
    at = dict(enumerate(rng.sample(list(rest), n)))
    others = [p for p in ports if p not in ("na.5", "na.9")]
    jobs = [(rng.randrange(n), rng.sample(ports, 2)) for _ in range(300)]
    jobs += [(v, ["na.5", "na.9"]) for v in range(3)] + [(3, ["na.9", others[0]]), (4, [others[1], "na.5"])]
    for r, legs in jobs:
        steps, final = solve(cm, rest, hops, at, r, legs)
        replay(cm, rest, at, r, legs, steps, final)
        at = final


def test_two_ports_in_one_cell_are_served_in_one_visit(kairos):  # Review Focus 1
    _, cm = kairos
    rest = rest_cells(cm, "branch")
    steps, _ = solve(cm, rest, station_hops(cm, len(cm.cells) - 26), dict(enumerate(rest)), 0, ["na.5", "na.9"])
    i, j = steps.index(("load", 0, "na.5")), steps.index(("load", 0, "na.9"))
    assert i < j and not any(s[0] == "move" and s[1] == 0 for s in steps[i:j])
