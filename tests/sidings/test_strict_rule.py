import hashlib
import json
import random

import pytest

from agvtb.sidings.strict_rule import (
    adjacency, pairs_ok_slow, rule_ok, strict_sidings, strict_sidings_of_layout, strict_sidings_of_world,
)
from helpers import DAIM, GRID_MAP, world_of

GOLDEN = DAIM / "tests" / "data" / "track_sidings_golden.json"
GOLDEN_SHA256 = "ef8578fd0c71a7b8e3a265ba94b23ea39c4355099f4758711f575fbe92d2437c"

def golden_cases():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]

def test_golden_file_is_the_checked_one():
    assert hashlib.sha256(GOLDEN.read_bytes()).hexdigest() == GOLDEN_SHA256

@pytest.mark.parametrize("case", golden_cases(), ids=lambda c: c["name"])
def test_matches_the_legacy_goldens(case):  # DAIM tests/test_track_sidings.py:28-30
    assert strict_sidings_of_layout(case["layout"]) == case["expected"]

def test_goldens_are_not_trivial():  # DAIM :33
    cases = golden_cases()
    assert len(cases) == 40 and sum(1 for c in cases if c["expected"]) >= 20
    assert sum(1 for c in cases if not c["layout"]["ports"]) == 5 and any(len(c["expected"]) >= 4 for c in cases)

def test_fast_rule_equals_pairwise_rule():  # DAIM :41-70, same generator and seed
    rng = random.Random(3)
    n_true = 0
    for _ in range(400):
        n = rng.randint(4, 14)
        nodes = list(range(n))
        arcs = set()
        for a in range(n - 1):
            r = rng.random()
            if r < 0.5:
                arcs |= {(a, a + 1), (a + 1, a)}
            elif r < 0.75:
                arcs.add((a, a + 1))
            else:
                arcs.add((a + 1, a))
        for _k in range(n):
            a, b = rng.sample(nodes, 2)
            arcs.add((a, b))
            if rng.random() < 0.5:
                arcs.add((b, a))
        ports = set(rng.sample(nodes, rng.randint(1, min(4, n - 1))))
        rest = [x for x in nodes if x not in ports]
        sidings = set(rng.sample(rest, rng.randint(0, len(rest))))
        _all, succ, pred = adjacency(nodes, arcs)
        fast, slow = rule_ok(succ, pred, ports, sidings), pairs_ok_slow(succ, ports, sidings)
        assert fast == slow, (sorted(arcs), ports, sidings)
        n_true += fast
    assert 20 < n_true < 380

# one-way ring 1->2->3->4->1, ports 1 and 3; two-way pockets 5 (at 2) and 6 (at 4); 7 can only be entered from 4
RING = ([1, 2, 3, 4, 5, 6, 7], [(1, 2), (2, 3), (3, 4), (4, 1), (2, 5), (5, 2), (4, 6), (6, 4), (4, 7)], [1, 3])

def test_ring_picks_only_the_two_way_pockets():  # DAIM :72-110
    nodes, arcs, ports = RING
    assert strict_sidings(nodes, arcs, ports) == ([5, 6], True)
    assert [strict_sidings(nodes, arcs, ports, target=t)[0] for t in (1, 0, 9)] == [[5], [], [5, 6]]
    assert strict_sidings(nodes, arcs, ports, fixed=[6], target=1) == ([6], True)
    assert strict_sidings(nodes, arcs, ports, fixed=[6], target=2) == ([5, 6], True)
    assert strict_sidings(nodes, arcs, ports, fixed=[2]) == ([2], False)
    assert strict_sidings(nodes, arcs, ports, fixed=[1, 99]) == ([5, 6], True)
    layout = {"nodes": [{"id": str(n)} for n in nodes],
              "segments": [{"id": i + 1, "startNodeId": str(a), "endNodeId": str(b)} for i, (a, b) in enumerate(arcs)],
              "ports": [{"id": "PA", "nodeId": "1"}, {"id": "PB", "nodeId": "3"}]}
    assert strict_sidings_of_layout(layout) == ["5", "6"]

def test_case_one_way_map():  # strict_expected.out.txt: 12 ids, equal to DAIM's stored list
    layout = json.loads((DAIM / "Maps" / "json" / "songdo_CASE_v0_xMSmap.json").read_text(encoding="utf-8"))
    expected = json.loads((DAIM / "Maps" / "json" / "songdo_CASE_v0_xMSmap_sidings.json").read_text(encoding="utf-8"))
    assert strict_sidings_of_layout(layout) == expected and len(expected) == 12

def test_world_and_layout_entry_points_agree():  # KAIROS: 14 sidings (strict_expected.out.txt)
    world = world_of("KAIROS_AMR")
    raw = json.loads((GRID_MAP / "KAIROS_AMR" / "KAIROS_AMR.map.json").read_text(encoding="utf-8"))
    got = strict_sidings_of_world(world)
    assert got == strict_sidings_of_layout(raw) and len(got) == 14
    assert got == sorted(got) and not set(got) & set(world.port_nodes)

# Beyond the plan's tests: the adjacency contract, rules that only read their arguments, target on a World.

def test_adjacency_names_arc_only_nodes_and_gives_empty_sets():  # DAIM _adjacency, track_sidings.py:27-36
    all_nodes, succ, pred = adjacency(["a"], [("a", "b"), ("b", "c")])
    assert all_nodes == {"a", "b", "c"} and succ["a"] == {"b"} and pred["c"] == {"b"}
    assert succ["c"] == set() and pred["a"] == set()

def snapshot(neighbours):
    return {node: set(others) for node, others in neighbours.items()}

def test_rules_only_read_their_arguments():  # DAIM's own rule_ok adds keys to its defaultdicts
    nodes, arcs, _ports = RING  # node 7 has no arc out: a search that indexes succ[7] would add that key
    for ports in ({1, 3}, set()):  # with ports rule_ok takes its fast path; without, it hands over to pairs_ok_slow
        _all, succ, pred = adjacency(nodes, arcs)
        before = snapshot(succ), snapshot(pred), set(ports)
        for sidings in (set(), {5}, {5, 6}, {7}):
            kept = set(sidings)
            rule_ok(succ, pred, ports, sidings)
            pairs_ok_slow(succ, ports, sidings)
            assert sidings == kept
        assert (snapshot(succ), snapshot(pred), ports) == before

def test_world_target_limits_the_greedy():
    world = world_of("KAIROS_AMR")
    full = strict_sidings_of_world(world)
    got = strict_sidings_of_world(world, target=5)  # the first five the greedy takes, as the full run takes them
    assert len(got) == 5 and set(got) <= set(full) and got == sorted(got)
    assert strict_sidings_of_world(world, target=0) == [] and strict_sidings_of_world(world, target=99) == full
