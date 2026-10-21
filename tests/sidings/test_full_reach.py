import pytest

from agvtb.sidings.condition_m import check_condition_m
from agvtb.sidings.full_reach import paper_outside, round_trip_misses
from agvtb.sidings.lattice import grid_sidings_of_world
from agvtb.sidings.strict_rule import strict_sidings_of_world
from agvtb.world.loader import load_world
from helpers import CONFIGS, DAIM_CONFIGS, fake_room_with_spurs, pocket_world, spur_room_world

OTHERS = ("c00", "c01", "c10", "c11", "c13", "c20", "c21", "c30", "c31")


def test_a_node_behind_a_full_siding_is_missed_from_the_other_siding(tmp_path):  # gate_expected.out.txt 1.
    lw = spur_room_world(tmp_path)
    assert round_trip_misses(lw.graph, lw.conflicts, ["c32", "c12"]) == {"c12": (), "c32": ("c13",)}
    assert check_condition_m(lw.graph, lw.conflicts, lw.world.port_nodes, ["c12", "c32"], 2).ok  # M alone passes


def test_a_siding_whose_every_stop_is_blocked_misses_every_node(tmp_path):  # gate_expected.out.txt 4.
    lw = spur_room_world(tmp_path)
    a = {sid for sid, st in lw.graph.items() if st.kind == "S" and st.nodes == ("c12",)}
    b = {sid for sid, st in lw.graph.items() if st.kind == "S" and st.nodes == ("c32",)}
    conflicts = {sid: cs | (b if sid in a else a if sid in b else frozenset()) for sid, cs in lw.conflicts.items()}
    assert round_trip_misses(lw.graph, conflicts, ["c12", "c32"]) == {"c12": OTHERS, "c32": OTHERS}


def test_fake_room_with_spurs_misses_nothing():  # gate_expected.out.txt 2.
    graph, conflicts, _ports = fake_room_with_spurs()
    assert round_trip_misses(graph, conflicts, ["c03", "c23", "c43"]) == {"c03": (), "c23": (), "c43": ()}


def test_variant_ga_on_mini_misses_nothing():  # gate_expected.out.txt 3., R07 §4.1
    lw = load_world(CONFIGS / "mini_deadend.json")
    misses = round_trip_misses(lw.graph, lw.conflicts, grid_sidings_of_world(lw.world, "bridge", (0, 0)))
    assert len(misses) == 37 and not any(misses.values())


def test_strict_rule_on_manual_traps_node_539():  # gate_expected.out.txt 3., R07 §4.2 (a few seconds)
    lw = load_world(DAIM_CONFIGS / "grid_songdo_manual.json")
    cands = strict_sidings_of_world(lw.world)
    usable = check_condition_m(lw.graph, lw.conflicts, lw.world.port_nodes, cands, 0).sidings
    misses = round_trip_misses(lw.graph, lw.conflicts, usable)
    assert (len(misses), sum(bool(m) for m in misses.values())) == (538, 535)
    assert {node for m in misses.values() for node in m} == {"539"}
    assert [s for s, m in misses.items() if not m] == ["479", "538", "540"]  # R07_files/full_sidings_reach.out.txt:103


def test_paper_condition_sees_a_pocket_the_round_trips_do_not(tmp_path):  # paper_expected.out.txt 2., R07 §2.4
    lw = pocket_world(tmp_path)
    assert round_trip_misses(lw.graph, lw.conflicts, ["r1c1", "r1c3"]) == {"r1c1": (), "r1c3": ()}
    assert paper_outside(lw.graph, lw.conflicts, lw.world.port_nodes, ["r1c1", "r1c3"]) == ("r2c1", "r2c2", "r2c3")


def test_paper_condition_on_the_other_scenes(tmp_path):  # paper_expected.out.txt 1.-3., R07 §4.3
    lw = spur_room_world(tmp_path)
    assert paper_outside(lw.graph, lw.conflicts, lw.world.port_nodes, ["c12", "c32"]) == ("c13",)
    graph, conflicts, ports = fake_room_with_spurs()
    assert paper_outside(graph, conflicts, ports, ["c03", "c23", "c43"]) == ()
    kairos = load_world(CONFIGS / "KAIROS_AMR.json")
    strict = strict_sidings_of_world(kairos.world)
    usable = check_condition_m(kairos.graph, kairos.conflicts, kairos.world.port_nodes, strict, 0).sidings
    assert paper_outside(kairos.graph, kairos.conflicts, kairos.world.port_nodes, usable) == (
        "na.191", "na.199", "na.200", "na.33", "na.4", "na.5", "na.6", "na.7", "na.8")


def test_sidings_must_be_nodes_of_the_graph():  # Claude's call 10
    graph, conflicts, ports = fake_room_with_spurs()
    with pytest.raises(ValueError, match="zz"):
        round_trip_misses(graph, conflicts, ["c03", "zz"])
    with pytest.raises(ValueError, match="zz"):
        paper_outside(graph, conflicts, ports, ["c03", "zz"])
