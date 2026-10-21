import pytest

from agvtb.planning.gate import GateError, check_gate
from agvtb.sidings.choice import candidates_of, parse_siding_choice
from agvtb.world.loader import load_world
from helpers import CONFIGS, fake_room_with_spurs, pocket_world, spur_room_world


def gate_of(lw, candidates, n):
    return check_gate(lw.graph, lw.conflicts, lw.world.port_nodes, candidates, n)


def test_gate_refuses_a_node_missed_behind_a_full_siding(tmp_path):  # user decision G2; gate_expected.out.txt 1.
    rep = gate_of(spur_room_world(tmp_path), ["c12", "c32"], 2)
    assert rep.condition_m.ok and not rep.ok
    assert rep.violations == ("siding c32: no round trip to nodes ['c13']",)
    assert rep.summary()["misses"] == {"c32": ["c13"]} and rep.summary()["ok"] is False
    assert rep.paper_outside == ("c13",)  # paper_expected.out.txt 1.


def test_gate_reports_the_paper_condition_without_judging_it(tmp_path):  # user decision G2; paper_expected.out.txt 2.
    rep = gate_of(pocket_world(tmp_path), ["r1c1", "r1c3"], 2)
    assert rep.ok and rep.violations == () and rep.paper_outside == ("r2c1", "r2c2", "r2c3")
    assert rep.summary()["paper_outside"] == ["r2c1", "r2c2", "r2c3"] and rep.summary()["ok"] is True


def test_fewer_usable_sidings_than_vehicles_is_a_violation(tmp_path):  # user decision G1
    rep = gate_of(spur_room_world(tmp_path), ["c12", "c32"], 3)
    assert rep.condition_m.fatal and rep.n_vehicles == 3
    assert rep.violations == ("only 2 usable sidings for 3 vehicles", "siding c32: no round trip to nodes ['c13']")


def test_variant_ga_passes_on_mini_up_to_its_37_sidings():  # facts.out.txt (c)
    lw = load_world(CONFIGS / "mini_deadend.json")
    cands = candidates_of(lw, parse_siding_choice("grid-bridge-s00"))
    assert gate_of(lw, cands, 37).ok and not gate_of(lw, cands, 38).ok


def test_kairos_strict_is_refused_and_variant_ga_passes():  # gate_expected.out.txt 3., R06 §4.3
    lw = load_world(CONFIGS / "KAIROS_AMR.json")
    strict = gate_of(lw, candidates_of(lw, parse_siding_choice("strict")), 1)
    assert not strict.ok and len(strict.condition_m.violations) == 13 and len(strict.paper_outside) == 9
    ga = gate_of(lw, candidates_of(lw, parse_siding_choice("strict-drop:na.34,na.53")), 12)
    assert ga.ok and ga.paper_outside == ()  # paper_expected.out.txt 3.


def test_fake_room_passes():
    graph, conflicts, ports = fake_room_with_spurs()
    rep = check_gate(graph, conflicts, ports, ["c03", "c23", "c43"], 3)
    assert rep.ok and rep.sidings == ("c03", "c23", "c43") and rep.violations == ()


def test_gate_error_carries_the_report(tmp_path):
    rep = gate_of(spur_room_world(tmp_path), ["c12", "c32"], 2)
    err = GateError(rep)
    assert err.report is rep and "c13" in str(err)


def test_rejects_fewer_than_one_vehicle(tmp_path):
    with pytest.raises(ValueError, match=r"\b0\b"):
        gate_of(spur_room_world(tmp_path), ["c12", "c32"], 0)
