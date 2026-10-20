import json
import os
import subprocess
import sys

import pytest

from agvtb.sidings.condition_m import EXCLUSION_REASONS, check_condition_m, usable_sidings
from agvtb.world.loader import load_world
from helpers import CONFIGS, DAIM_CONFIGS, daim_map_info, link_config

# DAIM tests/_fake_state_graph.py names nodes 'c<x><y>'; here the same names sit at (x, y) metres
def room(w, h):
    out = []
    for x in range(w):
        for y in range(h):
            if x + 1 < w:
                out.append((f"c{x}{y}", f"c{x + 1}{y}"))
            if y + 1 < h:
                out.append((f"c{x}{y}", f"c{x}{y + 1}"))
    return out

def loaded(tmp_path, links, ports, name):
    nodes = {n: (float(n[1]), float(n[2])) for link in links for n in link}
    return load_world(link_config(tmp_path, nodes, links, ports, name))

def check(lw, candidates, n):
    return check_condition_m(lw.graph, lw.conflicts, lw.world.port_nodes, candidates, n)

SPURS = room(6, 3) + [("c02", "c03"), ("c22", "c23"), ("c42", "c43")]
E3 = room(4, 2) + [("c11", "c12"), ("c12", "c13"), ("c31", "c32")]
E7 = [("c00", "c10"), ("c00", "c01"), ("c10", "c11"), ("c01", "c11"), ("c10", "c20"), ("c20", "c30"), ("c30", "c40"),
      ("c40", "c50"), ("c40", "c41"), ("c50", "c51"), ("c41", "c51"), ("c01", "c02"), ("c51", "c52")]

def test_room_with_spurs(tmp_path):  # DAIM tests/test_safe_tail_map.py:23 and :77
    rep = check(loaded(tmp_path, SPURS, ["c00", "c50"], "spurs"), ["c03", "c23", "c43"], 2)
    assert rep.summary() == {"ok": True, "fatal": False, "sidings": 3, "n_vehicles": 2, "excluded": {},
                             "aisle_size": 18, "violations": []}
    assert rep.sidings == ("c03", "c23", "c43")

def test_too_few_sidings_is_fatal(tmp_path):  # :30
    rep = check(loaded(tmp_path, SPURS, ["c00", "c50"], "spurs"), ["c03", "c23", "c43"], 4)
    assert rep.fatal and not rep.ok

def test_only_access_to_a_port_is_excluded(tmp_path):  # :35
    rep = check(loaded(tmp_path, E3, ["c00", "c13"], "access"), ["c12", "c32"], 1)
    assert (rep.excluded, rep.sidings, rep.ok, rep.aisle_size) == ({"c12": "port_access"}, ("c32",), True, 10)

def test_reachable_only_through_a_port_is_a_pocket(tmp_path):  # :44
    rep = check(loaded(tmp_path, E3, ["c00", "c12"], "pocket"), ["c13", "c32"], 1)
    assert (rep.excluded, rep.sidings, rep.ok) == ({"c13": "pocket"}, ("c32",), True)

def test_port_and_unknown_candidates(tmp_path):  # :51
    lw = loaded(tmp_path, SPURS, ["c00", "c50"], "spurs")
    usable, excluded, _aisle = usable_sidings(lw.graph, lw.world.port_nodes, ["c03", "c00", "zz"])
    assert (usable, excluded) == (["c03"], {"c00": "port", "zz": "not_in_graph"})
    assert set(excluded.values()) <= set(EXCLUSION_REASONS)

def test_port_outside_the_aisle(tmp_path):  # :56
    rep = check(loaded(tmp_path, SPURS + [("c90", "c91")], ["c00", "c50", "c90"], "outside"), ["c03", "c23", "c43"], 2)
    assert not rep.fatal and not rep.ok and any("c90" in v for v in rep.violations)

def test_footprint_that_cuts_the_aisle(tmp_path):  # :64 — a vehicle parked at c02 is made to cover the aisle c20-c30
    lw = loaded(tmp_path, E7, ["c00", "c50"], "patch")
    assert check(lw, ["c02", "c52"], 2).ok
    stops = {sid for sid, st in lw.graph.items() if st.kind == "S" and st.nodes == ("c02",)}
    cut = {"M,c20,c30", "M,c30,c20"}
    conflicts = dict(lw.conflicts)
    for sid in stops:
        conflicts[sid] = conflicts[sid] | cut
    for sid in cut:
        conflicts[sid] = conflicts[sid] | stops
    rep = check_condition_m(lw.graph, conflicts, lw.world.port_nodes, ["c02", "c52"], 2)
    assert len(rep.violations) == 1 and "c52" in rep.violations[0] and "c00" in rep.violations[0]

def test_rejects_negative_vehicles(tmp_path):
    with pytest.raises(ValueError, match="-1"):
        check(loaded(tmp_path, SPURS, ["c00", "c50"], "spurs"), ["c03"], -1)

DAIM_RECORDS = [  # (config, N, usable, excluded, aisle); condm_prototype.out.txt reproduces both on our graph
    # MAPF/DAIM_simulation/docs/superpowers/plans/2026-10-06-safe-tail.reports/check_M_result_2026-10-06.jsonl:1
    ("grid_songdo_travelbr", 15, 195, {}, 728),
    # R/experiments/results/songdo_placement_engines/sim_travel_s*.json, rule_stats.safe_tail_map (misc_facts.out.txt (e))
    ("grid_songdo_travel", 15, 70, {}, 853),
]

@pytest.mark.parametrize("name, n, usable, excluded, aisle", DAIM_RECORDS)
def test_daim_records(name, n, usable, excluded, aisle):
    lw = load_world(DAIM_CONFIGS / f"{name}.json")
    rep = check(lw, [str(s) for s in daim_map_info(name)["sidings"]], n)
    assert (len(rep.sidings), rep.excluded, rep.aisle_size, rep.ok) == (usable, excluded, aisle, True)

def test_results_ignore_hash_seed():
    code = (
        "import json\n"
        "from agvtb.world.loader import load_world\n"
        "from agvtb.sidings.strict_rule import strict_sidings_of_world\n"
        "from agvtb.sidings.condition_m import check_condition_m\n"
        f"lw = load_world({str(CONFIGS / 'mini_deadend.json')!r})\n"
        "s = strict_sidings_of_world(lw.world)\n"
        "r = check_condition_m(lw.graph, lw.conflicts, lw.world.port_nodes, s, 0)\n"
        "print(json.dumps([s, r.summary(), list(r.sidings)]))\n"
    )
    outs = {subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed},
                           capture_output=True, text=True, check=True, timeout=300).stdout for seed in ("1", "2")}
    assert len(outs) == 1

# Beyond the plan's tests: rules the brief names that its tests do not pin. DAIM's own check_safe_tail_map
# (core/safe_tail_map.py), run on the same graph and conflicts, gives the same result on each case below,
# apart from the wording of its messages.

def one_way(tmp_path, links, ports, name, closed):  # `closed`: directed segments (u, v) taken out of the two-way map
    nodes = {n: (float(n[1]), float(n[2])) for link in links for n in link}
    cfg = link_config(tmp_path, nodes, links, ports, name)
    path = tmp_path / f"{name}_map.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["segments"] = [s for s in data["segments"] if (s["startNodeId"], s["endNodeId"]) not in closed]
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_world(cfg)

def stops_at(lw, node):
    return {sid for sid, st in lw.graph.items() if st.kind == "S" and st.nodes == (node,)}

def test_every_port_is_seen_before_the_aisle_is_found_again(tmp_path):  # :101-128, not once per port
    # c02 reaches the aisle c00-c30 through c01, its neighbour c12 through c11; the aisle is found once per round
    links = [("c02", "c12"), ("c02", "c01"), ("c01", "c00"), ("c12", "c11"), ("c11", "c10"), ("c00", "c10"),
             ("c10", "c20"), ("c20", "c30")]
    lw = loaded(tmp_path, links, ["c02", "c12"], "rounds")
    usable, excluded, aisle = usable_sidings(lw.graph, lw.world.port_nodes, ["c01", "c11"])
    assert (usable, excluded, len(aisle)) == ([], {"c01": "port_access", "c11": "port_access"}, 8)

def test_tied_components_make_the_one_with_the_smallest_node_the_aisle(tmp_path):  # :67-84
    links = [("c00", "c10"), ("c10", "c20"), ("c20", "c30"), ("c30", "c40")]  # c20 leaves two components of 2 nodes
    near = loaded(tmp_path, links, ["c00"], "near")  # the port is in the aisle: c20 stays
    usable, excluded, aisle = usable_sidings(near.graph, near.world.port_nodes, ["c20"])
    assert (usable, excluded, aisle) == (["c20"], {}, frozenset({"c00", "c10"}))
    far = loaded(tmp_path, links, ["c40"], "far")  # the port is in the other component: c20 is its access
    usable, excluded, aisle = usable_sidings(far.graph, far.world.port_nodes, ["c20"])
    assert (usable, excluded, len(aisle)) == ([], {"c20": "port_access"}, 5)

def test_the_way_to_the_aisle_goes_through_the_smaller_neighbour(tmp_path):  # :107-126, neighbours in sorted order
    # port c21 has two ways of equal length to the aisle {c00, c10}: through c11 and through c20
    links = [("c10", "c20"), ("c20", "c21"), ("c21", "c11"), ("c11", "c10"), ("c10", "c00")]
    lw = loaded(tmp_path, links, ["c21"], "square")
    usable, excluded, _aisle = usable_sidings(lw.graph, lw.world.port_nodes, ["c20", "c11"])
    assert (usable, excluded) == (["c20"], {"c11": "port_access"})

def test_pockets_are_looked_for_only_in_a_round_without_port_access(tmp_path):  # :129-135 after :127-128
    # c30 is the access of port c40. c60 touches the aisle only through c50, which joins it once c30 is dropped
    links = [("c00", "c10"), ("c10", "c20"), ("c20", "c30"), ("c30", "c40"), ("c40", "c50"), ("c50", "c60")]
    lw = loaded(tmp_path, links, ["c40"], "line")
    usable, excluded, aisle = usable_sidings(lw.graph, lw.world.port_nodes, ["c30", "c60"])
    assert (usable, excluded, len(aisle)) == (["c60"], {"c30": "port_access"}, 6)

def test_a_candidate_that_is_a_port_but_no_node_is_not_in_graph(tmp_path):  # :95-98, not_in_graph is asked first
    lw = loaded(tmp_path, SPURS, ["c00", "c50"], "spurs")
    _usable, excluded, _aisle = usable_sidings(lw.graph, ["c00", "c50", "zz"], ["zz", "c00"])
    assert excluded == {"zz": "not_in_graph", "c00": "port"}

def test_the_round_trip_needs_both_ways(tmp_path):  # :196-198, a port pose in the forward and in the backward set
    # hub c10 and port c20 are linked both ways; c00 -> c10 can only be left, c10 -> c11 can only be entered
    lw = one_way(tmp_path, [("c00", "c10"), ("c10", "c11"), ("c10", "c20")], ["c20"], "hub",
                 {("c10", "c00"), ("c11", "c10")})
    rep = check(lw, ["c00", "c11"], 2)
    assert rep.sidings == ("c00", "c11") and len(rep.violations) == 2
    assert "c00" in rep.violations[0] and "c11" in rep.violations[1] and all("c20" in v for v in rep.violations)

def test_violations_come_in_the_documented_order(tmp_path):  # ports outside the aisle, ports without a load pose, sidings
    lw = loaded(tmp_path, SPURS, ["c00", "c50"], "spurs")
    rep = check_condition_m(lw.graph, lw.conflicts, ["zz", "c50", "yy", "c00"], ["c03", "c23", "c43"], 2)
    # the two ports that are no nodes are outside the aisle and have no load pose; the other ports and the sidings pass
    assert [v.split()[1] for v in rep.violations] == ["yy", "zz", "yy", "zz"]  # each line starts "port <id>"

def test_siding_whose_stop_poses_are_all_blocked(tmp_path):  # :192-194
    lw = loaded(tmp_path, SPURS, ["c00", "c50"], "spurs")
    first, second = stops_at(lw, "c03"), stops_at(lw, "c23")  # a vehicle at either one covers every pose of the other
    conflicts = dict(lw.conflicts)
    for sid in first:
        conflicts[sid] = conflicts[sid] | second
    for sid in second:
        conflicts[sid] = conflicts[sid] | first
    rep = check_condition_m(lw.graph, conflicts, lw.world.port_nodes, ["c03", "c23", "c43"], 2)
    assert len(rep.violations) == 2 and "c03" in rep.violations[0] and "c23" in rep.violations[1]

def test_a_parked_vehicle_blocks_the_states_that_take_its_node_whatever_the_conflicts(tmp_path):  # :165-174
    # one-way ring c10 -> c20 -> c21 -> c11 -> c10, port c11, and c00 linked both ways to c10. The only way from
    # c00 to the port runs through the siding c21; the conflicts may list nothing, a vehicle at c21 still cuts it
    links = [("c10", "c20"), ("c20", "c21"), ("c21", "c11"), ("c11", "c10"), ("c00", "c10")]
    lw = one_way(tmp_path, links, ["c11"], "ring", {("c20", "c10"), ("c21", "c20"), ("c11", "c21"), ("c10", "c11")})
    nothing = {sid: frozenset() for sid in lw.graph}
    for conflicts in (lw.conflicts, nothing):
        rep = check_condition_m(lw.graph, conflicts, lw.world.port_nodes, ["c00", "c21"], 2)
        assert len(rep.violations) == 1 and "c00" in rep.violations[0] and "c11" in rep.violations[0]

def test_ports_and_candidates_may_be_any_iterables_with_repeats(tmp_path):
    lw = loaded(tmp_path, SPURS, ["c00", "c50"], "spurs")
    ports = iter(["c50", "zz", "c00", "zz"])  # zz is no node: outside the aisle and without a load pose, once each
    candidates = (c for c in ["c43", "c03", "c23", "c03", "c99", "c99"])
    rep = check_condition_m(lw.graph, lw.conflicts, ports, candidates, 3)
    assert (rep.sidings, rep.excluded, len(rep.violations)) == (("c03", "c23", "c43"), {"c99": "not_in_graph"}, 2)
