import itertools
import json
import re

import pytest

from agvtb.sidings.grid_rule import GRID_MODES, CellGrid, grid_candidates, grid_sidings
from agvtb.sidings.lattice import GRID_SHIFTS, LatticeError, grid_sidings_of_world, lattice_grid
from agvtb.world.config import load_map_config
from agvtb.world.loader import load_world
from helpers import DAIM_CONFIGS, daim_map_info, daim_world, link_config, world_of, world_of_config

D12_RECORDED = [  # d12_world_route.out.txt: our World -> lattice -> rule gives the map's own info.sidings, same order
    ("grid_songdo_travelbr", "bridge"), ("grid_songdo_dflpbr", "bridge"), ("grid_songdo_greedybr", "bridge"),
    ("grid_songdo_rand", "bridge"), ("grid_songdo_manual", "bridge"),
    ("grid_songdo_travel", "legacy"), ("grid_songdo_dflp", "legacy"), ("grid_songdo_greedy", "legacy"),
    ("grid_songdo_joint", "bridge"), ("grid_songdo_joint", "legacy"),
]

@pytest.mark.parametrize("name, mode", D12_RECORDED)
def test_daim_maps_give_their_recorded_sidings(name, mode):
    assert grid_sidings_of_world(daim_world(name), mode) == [str(s) for s in daim_map_info(name)["sidings"]]

def test_daim_configs_match_their_maps():
    for path in sorted(DAIM_CONFIGS.glob("*.json")):
        cfg = load_map_config(path)
        kin = json.loads(cfg.map_path.read_text(encoding="utf-8"))["vehicleModels"][0]["kinematics"]
        assert (cfg.accel_mps2, cfg.decel_mps2) == (kin["accel"] / 1000, kin["decel"] / 1000)
        assert (cfg.rotation_s_per_90deg, cfg.dwell_s) == (90 / kin["angularSpeed"], kin["dwell"])
        assert load_world(path).world.collision == "grid"  # the cell rule holds: diagonal 1.0 m = pitch
    assert len(list(DAIM_CONFIGS.glob("*.json"))) == 9

SONGDO_ACCESS = {"Port1": "G28_41", "Port2": "G44_41", "Port3": "G46_32", "Port4": "G46_26", "Port5": "G33_19",
                 "Port6": "G83_47", "Port7": "G83_40", "Port8": "G81_33", "Port9": "G81_26", "Port10": "G114_21"}

def test_lattice_of_the_three_px_maps():  # lattice_adjacency.out.txt
    mini, rnd, songdo = (lattice_grid(world_of(n)) for n in ("mini_deadend", "random_32_32_20", "songdo_grid_bi"))
    assert (mini.grid.rows, mini.grid.cols, len(mini.cell_of), len(mini.grid.ports), mini.access) == (10, 25, 166, 16, {})
    assert (rnd.grid.rows, rnd.grid.cols, len(rnd.cell_of), len(rnd.grid.ports), rnd.access) == (34, 34, 846, 47, {})
    assert (songdo.grid.rows, songdo.grid.cols, len(songdo.cell_of), songdo.pitch_m) == (51, 107, 1376, 1.0)
    assert songdo.access == SONGDO_ACCESS and len(songdo.grid.ports) == 10
    assert all(songdo.cell_of[a] in songdo.grid.ports for a in SONGDO_ACCESS.values())

COUNTS = [  # (map, shift, candidates, bridge, legacy): DAIM's auto_parks on the same grids, grid_rule_expected.out.txt (U2: access cells)
    ("mini_deadend", (0, 0), 39, 37, 36), ("mini_deadend", (0, 1), 36, 36, 36),
    ("mini_deadend", (1, 0), 39, 37, 35), ("mini_deadend", (1, 1), 36, 36, 36),
    ("random_32_32_20", (0, 0), 197, 146, 137), ("random_32_32_20", (0, 1), 204, 157, 144),
    ("random_32_32_20", (1, 0), 197, 156, 134), ("random_32_32_20", (1, 1), 201, 154, 135),
    ("songdo_grid_bi", (0, 0), 335, 333, 326), ("songdo_grid_bi", (0, 1), 351, 348, 236),
    ("songdo_grid_bi", (1, 0), 332, 331, 331), ("songdo_grid_bi", (1, 1), 348, 347, 346),
]

@pytest.mark.parametrize("name, shift, candidates, bridge, legacy", COUNTS)
def test_counts_by_shift(name, shift, candidates, bridge, legacy):
    world = world_of(name)
    assert len(grid_candidates(lattice_grid(world, shift).grid)) == candidates
    assert len(grid_sidings_of_world(world, "bridge", shift)) == bridge
    assert len(grid_sidings_of_world(world, "legacy", shift)) == legacy

def test_sidings_are_lattice_nodes_and_never_ports_or_access_nodes():
    # not trivially true: at shift (0, 0) the access nodes of Port7 and Port9 sit on (odd, odd) cells
    world = world_of("songdo_grid_bi")
    got = set(grid_sidings_of_world(world))
    assert got <= set(lattice_grid(world).cell_of)
    assert not got & (set(SONGDO_ACCESS.values()) | set(world.port_nodes))

def test_kairos_is_not_a_lattice():
    # 59 of its 61 nodes are off the 1.501 m lattice of na.44, 46 of them no port. The message says so; it does not blame
    # the first port it meets (na.1, linked to na.2, which is off the lattice too). na.2 is the first of the 46 in the map file
    expected = r"^KAIROS_AMR: 59 of 61 nodes are off the 1\.501 m lattice of na\.44 \(46 are no port\), first na\.2: "
    with pytest.raises(LatticeError, match=expected):
        lattice_grid(world_of("KAIROS_AMR"))

# a 1 m line L0-L1-L2-L3: its six 1 m segments outnumber the extra links in every case, so the pitch stays 1 m
LINE = {"L0": (0.0, 0.0), "L1": (1.0, 0.0), "L2": (2.0, 0.0), "L3": (3.0, 0.0)}
LINE_LINKS = [("L0", "L1"), ("L1", "L2"), ("L2", "L3")]

@pytest.mark.parametrize("extra_nodes, extra_links, ports, names", [
    ({"X1": (1.0, 0.4)}, [("L1", "X1")], ["L0"], ["X1"]),                       # off the lattice and not a port
    ({"X1": (0.5, 0.4)}, [("L0", "X1"), ("L1", "X1")], ["X1"], ["X1"]),         # off-lattice port with two links
    ({"U1": (0.0, 1.0), "W1": (1.0, 1.0)}, [("L0", "U1"), ("L1", "W1")], ["L0"], ["U1", "W1"]),  # U1, W1 not linked
    ({"L0B": (0.0, 0.0)}, [("L0B", "L1")], ["L3"], ["L0", "L0B"]),               # two nodes on one lattice point
])
def test_rejects_bad_lattices(tmp_path, extra_nodes, extra_links, ports, names):
    world = world_of_config(link_config(tmp_path, {**LINE, **extra_nodes}, LINE_LINKS + extra_links, ports))
    with pytest.raises(LatticeError) as err:
        lattice_grid(world)
    assert all(re.search(rf"\b{name}\b", str(err.value)) for name in names)  # whole ids: L0 must not match inside L0B

def test_many_nodes_off_the_lattice_are_counted_and_checked_before_the_ports(tmp_path):
    # Only a port may be off the lattice. P1, the first node off it in the map file, is a port with two links, a fault too;
    # but X3, X1 and X2 are no ports. The message counts the nodes off the lattice and those that are no port, and names X3,
    # the first of the three in the file (X1 is the first by id), not P1: the ports are looked at after the nodes that are no port
    nodes = {**LINE, "P1": (2.5, 0.7), "X3": (0.5, 0.4), "X1": (1.5, 0.4), "X2": (3.5, 0.4)}
    world = world_of_config(link_config(tmp_path, nodes, LINE_LINKS + [("L2", "P1"), ("L3", "P1")], ["L0", "P1"]))
    with pytest.raises(LatticeError) as err:
        lattice_grid(world)
    assert str(err.value).startswith("links: 4 of 8 nodes are off the 1 m lattice of L0 (3 are no port), first X3: ")

def test_rejects_bad_shift():
    with pytest.raises(ValueError, match=r"\(2, 0\)"):
        lattice_grid(world_of("mini_deadend"), (2, 0))
    assert GRID_SHIFTS == ((0, 0), (0, 1), (1, 0), (1, 1))

# Rules the cases above leave open (added with the implementation; not in the plan's list)
@pytest.mark.parametrize("extra_nodes, extra_links, ports, names", [
    ({"S1": (0.0, 2.0)}, [("L0", "S1")], ["L3"], ["L0", "S1"]),                          # a segment that skips a cell
    ({"D1": (1.0, 1.0)}, [("L0", "D1")], ["L3"], ["L0", "D1"]),                          # a diagonal segment
    ({"X1": (0.5, 0.4), "X2": (0.5, 0.8)}, [("X1", "X2")], ["X1", "X2"], ["X1", "X2"]),  # a port linked only to a node off the lattice
])
def test_rejects_more_bad_lattices(tmp_path, extra_nodes, extra_links, ports, names):
    world = world_of_config(link_config(tmp_path, {**LINE, **extra_nodes}, LINE_LINKS + extra_links, ports))
    with pytest.raises(LatticeError) as err:
        lattice_grid(world)
    assert all(re.search(rf"\b{name}\b", str(err.value)) for name in names)

def test_rejects_a_world_without_segments(tmp_path):
    import dataclasses
    world = dataclasses.replace(world_of_config(link_config(tmp_path, LINE, LINE_LINKS, ["L0"])), segments={})
    with pytest.raises(LatticeError, match="links"):
        lattice_grid(world)

def test_ports_follow_port_nodes_and_each_cell_comes_once(tmp_path):
    # N3..N0 sit at x = 0..3, so the sorted port nodes run against the cell order; X1 and X2 are ports off the lattice at N1
    nodes = {"N3": (0.0, 0.0), "N2": (1.0, 0.0), "N1": (2.0, 0.0), "N0": (3.0, 0.0), "X1": (2.0, 0.4), "X2": (2.0, -0.6)}
    links = [("N3", "N2"), ("N2", "N1"), ("N1", "N0"), ("N1", "X1"), ("N1", "X2")]
    world = world_of_config(link_config(tmp_path, nodes, links, ["N1", "X2", "X1", "N0"]))
    lattice = lattice_grid(world)
    assert world.port_nodes == ("N0", "N1", "X1", "X2")
    assert lattice.access == {"X1": "N1", "X2": "N1"}
    assert lattice.grid.ports == ((0, 3), (0, 2))  # N0, then N1 once for N1, X1 and X2
    assert (lattice.grid.rows, lattice.grid.cols) == (1, 4) and lattice.cell_of["N3"] == (0, 0)  # X2 below adds no row
    assert lattice.node_at == {cell: node for node, cell in lattice.cell_of.items()}

def test_pitch_tie_goes_to_the_shorter_length(tmp_path):
    # two 1 m segments and two 2.5 m ones: the pitch is 1 m, and P, off that lattice, hangs on L0
    nodes = {"L0": (0.0, 0.0), "L1": (1.0, 0.0), "P": (0.0, 2.5)}
    lattice = lattice_grid(world_of_config(link_config(tmp_path, nodes, [("L0", "L1"), ("L0", "P")], ["P"])))
    assert (lattice.pitch_m, lattice.access, lattice.grid.ports) == (1.0, {"P": "L0"}, ((0, 0),))

def test_pitch_is_the_real_length_of_the_reference_segment(tmp_path):
    # a 1.0000004 m step counts as 1 m, but then the fourth node would sit 1.2e-6 m off the lattice
    step = 1.0000004
    nodes = {f"L{k}": (k * step, 0.0) for k in range(5)}
    links = [(f"L{k}", f"L{k + 1}") for k in range(4)]
    lattice = lattice_grid(world_of_config(link_config(tmp_path, nodes, links, ["L0"])))
    assert lattice.pitch_m == pytest.approx(step, rel=1e-12) and len(lattice.cell_of) == 5

def test_bridge_is_the_default_mode_and_the_shift_is_kept_as_plain_ints():
    world = world_of("mini_deadend")  # bridge 37 sidings, legacy 36
    assert grid_sidings_of_world(world) == grid_sidings_of_world(world, "bridge") != grid_sidings_of_world(world, "legacy")
    assert lattice_grid(world, (False, True)).shift == (0, 1)  # equal to a member of GRID_SHIFTS, stored as that member
    assert all(type(n) is int for cell in lattice_grid(world, (False, True)).cell_of.values() for n in cell)

# Final review fix F3: what the lattice_grid docstring says of DAIM's numbering when its grid has blocked rows below or columns left.

@pytest.mark.parametrize("name", ["mini_deadend", "random_32_32_20"])
@pytest.mark.parametrize("mode", GRID_MODES)
def test_shift_by_the_parity_of_the_blocked_rows_and_columns_gives_daims_numbering(name, mode):
    # A DAIM grid may have `below` blocked rows under and `left` blocked columns left of the lattice nodes (and blocked margins
    # above and right). The grid rule reads a cell number only through its row and column parity, so the member of GRID_SHIFTS
    # equal to (below, left) mod 2 picks the same sidings as that numbering, which the grid rule is run on here directly
    world = world_of(name)
    lattice = lattice_grid(world)
    free = lattice.grid.free_cells()
    for below, left in itertools.product(range(4), repeat=2):
        rows, cols = lattice.grid.rows + below + 1, lattice.grid.cols + left + 2
        daim = CellGrid(
            rows, cols,
            frozenset((r, c) for r in range(rows) for c in range(cols) if (r - below, c - left) not in free),
            tuple((r + below, c + left) for r, c in lattice.grid.ports),
        )
        got = sorted(lattice.node_at[(r - below, c - left)] for r, c in grid_sidings(daim, mode))
        assert got == sorted(grid_sidings_of_world(world, mode, (below % 2, left % 2))), (below, left)
