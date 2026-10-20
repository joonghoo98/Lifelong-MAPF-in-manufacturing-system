import json

import pytest

from agvtb.sidings.grid_rule import GRID_MODES, CellGrid, grid_candidates, grid_sidings, load_daim_grid
from helpers import DAIM_ENGINES

def bottleneck():  # DAIM tests/test_grid_park_restore.py:21-25
    blocked = frozenset((r, c) for r in range(9) for c in (5, 6, 7) if r != 3)
    return CellGrid(9, 15, blocked, ((0, 0), (8, 0), (0, 4), (8, 4)))

def ports_connected_when_full(grid, parks):
    """One 4-neighbour component of the free cells minus the parks holds every port."""
    corridor = set(grid.free_cells()) - set(parks)
    start = grid.ports[0]
    seen, stack = {start}, [start]
    while stack:
        r, c = stack.pop()
        for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if nb in corridor and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return set(grid.ports) <= seen

def test_bottleneck_bridge_keeps_the_far_room():  # DAIM :36; exact lists from DAIM's own code (grid_rule_expected.out.txt)
    bridge, legacy = grid_sidings(bottleneck(), "bridge"), grid_sidings(bottleneck(), "legacy")
    assert bridge == [(r, c) for r in (1, 3, 5, 7) for c in (1, 3, 9, 11, 13)]
    assert legacy == [(r, c) for r in (1, 3, 5, 7) for c in (1, 3)]
    assert (3, 5) not in bridge and (3, 7) not in bridge and set(legacy) <= set(bridge)
    assert ports_connected_when_full(bottleneck(), bridge) and ports_connected_when_full(bottleneck(), legacy)
    assert grid_sidings(bottleneck()) == bridge  # bridge is the default

@pytest.mark.parametrize("mode", GRID_MODES)
def test_open_room(mode):  # DAIM :64-70
    room = CellGrid(4, 8, frozenset(), ((0, 0), (3, 7), (0, 7), (3, 0)))
    assert grid_sidings(room, mode) == [(1, 1), (1, 3), (1, 5), (1, 7), (3, 1), (3, 3), (3, 5)]

@pytest.mark.parametrize("mode", GRID_MODES)
def test_isolated_piece(mode):  # DAIM :73-81
    grid = CellGrid(5, 9, frozenset((r, 4) for r in range(5)), ((0, 0), (4, 0)))
    assert grid_sidings(grid, mode) == [(1, 1), (1, 3), (3, 1), (3, 3)]
    assert ports_connected_when_full(grid, grid_sidings(grid, mode))

def test_empty_grid():
    empty = CellGrid(0, 0, frozenset(), ())
    assert grid_sidings(empty) == [] and grid_candidates(empty) == []

def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="'Legacy'"):
        grid_sidings(bottleneck(), "Legacy")

def recorded_parks(name):
    data = json.loads((DAIM_ENGINES / "layouts" / f"{name}.grid.json").read_text(encoding="utf-8"))
    return sorted(tuple(p) for p in data["parks"])

RECORDED = [  # (layout grid, the mode that made its parks, count): verify_m2_facts.out.txt; joint: misc_facts.out.txt (f)
    ("songdo_travel", "legacy", 70), ("songdo_travelbr", "bridge", 195),
    ("songdo_dflp", "legacy", 18), ("songdo_dflpbr", "bridge", 192),
    ("songdo_greedy", "legacy", 195), ("songdo_greedybr", "bridge", 196),
    ("songdo_rand", "bridge", 212), ("songdo_rand", "legacy", 212),
    ("songdo_manual", "bridge", 214), ("songdo_manual", "legacy", 214),
    ("songdo_joint", "bridge", 199), ("songdo_joint", "legacy", 199),
]

@pytest.mark.parametrize("name, mode, count", RECORDED)
def test_daim_layout_grids(name, mode, count):
    parks = recorded_parks(name)
    assert grid_sidings(load_daim_grid(DAIM_ENGINES / "layouts" / f"{name}.grid.json"), mode) == parks
    assert len(parks) == count

@pytest.mark.parametrize("legacy_twin, bridge_twin", [
    ("songdo_travel", "songdo_travelbr"), ("songdo_dflp", "songdo_dflpbr"), ("songdo_greedy", "songdo_greedybr"),
])
def test_twin_grids_share_the_cells(legacy_twin, bridge_twin):
    # the twins differ only in name and parks, so bridge on the legacy twin gives the bridge twin's parks
    twin = load_daim_grid(DAIM_ENGINES / "layouts" / f"{legacy_twin}.grid.json")
    assert twin == load_daim_grid(DAIM_ENGINES / "layouts" / f"{bridge_twin}.grid.json")
    assert grid_sidings(twin, "bridge") == recorded_parks(bridge_twin)

def test_candidates_of_the_songdo_grids():  # (odd, odd) non-port free cells: verify_m2_facts.out.txt
    counts = [len(grid_candidates(load_daim_grid(DAIM_ENGINES / "layouts" / f"songdo_{n}.grid.json")))
              for n in ("travel", "dflp", "greedy", "rand", "manual")]
    assert counts == [209, 195, 198, 215, 214]

# --- tests beyond the plan's: details the tests above do not pin down ---

def test_free_cells():
    # the cells of the rows x cols block that are not blocked; a blocked cell outside the block is no cell of it
    grid = CellGrid(2, 3, frozenset({(0, 1), (5, 5)}), ())
    assert grid.free_cells() == {(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)}

def test_candidates_are_the_odd_odd_free_cells_that_are_no_port():
    # of the odd-odd cells, (1, 1) is blocked and (3, 3) is a port; a port on an even cell takes no candidate
    grid = CellGrid(4, 4, frozenset({(1, 1)}), ((3, 3), (0, 0), (2, 1)))
    assert grid_candidates(grid) == [(1, 3), (3, 1)]

def test_load_daim_grid_reads_four_keys_and_keeps_the_port_order(tmp_path):
    path = tmp_path / "tiny.grid.json"
    path.write_text(json.dumps({
        "name": "tiny", "pitch": 1000.0, "rows": 3, "cols": 4, "blocked": [[2, 3], [0, 1]],
        "ports": [[2, 0], [0, 0], [2, 0]], "parks": [[1, 1]], "origin": [0.0, 0.0],
    }), encoding="utf-8")
    grid = load_daim_grid(path)
    assert grid == CellGrid(3, 4, frozenset({(0, 1), (2, 3)}), ((2, 0), (0, 0), (2, 0)))  # ports: file order, no dedupe
    assert isinstance(grid.blocked, frozenset) and isinstance(grid.ports, tuple)
    assert load_daim_grid(str(path)) == grid

def test_unknown_mode_is_rejected_whatever_the_grid():
    with pytest.raises(ValueError, match=r"^grid rule mode must be one of bridge, legacy, got 'fast'$"):
        grid_sidings(CellGrid(0, 0, frozenset(), ()), "fast")

# The main component is the one with the most ports, then with the most cells, then the one that holds the smallest cell.
# The grids below are worked out by hand. DAIM leaves the last tie-break to set order, so its own result on a tie is no evidence.

@pytest.mark.parametrize("mode", GRID_MODES)
def test_main_component_has_the_most_cells_when_the_ports_tie(mode):
    # a wall on column 2 cuts the grid in two with 1 port each; the left piece holds (0, 0) but is only 2 columns wide
    grid = CellGrid(5, 9, frozenset((r, 2) for r in range(5)), ((0, 0), (0, 8)))
    assert grid_sidings(grid, mode) == [(r, c) for r in (1, 3) for c in (3, 5, 7)]

@pytest.mark.parametrize("mode", GRID_MODES)
def test_a_port_listed_twice_counts_once(mode):
    # a wall on column 3 cuts the grid in two: the left piece holds 2 ports, the bigger right piece 1 port listed twice
    grid = CellGrid(5, 9, frozenset((r, 3) for r in range(5)), ((0, 0), (4, 0), (0, 8), (0, 8)))
    assert grid_sidings(grid, mode) == [(1, 1), (3, 1)]  # counted twice, the right piece would win and keep its sidings

def test_bridge_tie_goes_to_the_component_with_the_smallest_cell():
    # ......   blocked (1, 4) (2, 2) (2, 4) (3, 0); ports (1, 2) (3, 3)
    # ..P.#.   The top and the bottom component both hold 1 port and 10 cells; the top one holds (0, 0).
    # ..#.#.   The lone cell (2, 5) is joined through (1, 5), which touches the top one, so (3, 5) stays a siding.
    # #..P..   (With the bottom one as the main component it would be joined through (3, 5), and (1, 5) would stay.)
    # ......
    grid = CellGrid(5, 6, frozenset({(1, 4), (2, 2), (2, 4), (3, 0)}), ((1, 2), (3, 3)))
    assert grid_sidings(grid, "bridge") == [(1, 1), (3, 5)]

def test_legacy_tie_goes_to_the_component_with_the_smallest_cell():
    # ...P.   blocked (1, 4) (2, 2) (3, 0); ports (0, 3) (3, 4)
    # ....#   The top and the bottom component both hold 1 port and 9 cells; the top one holds (0, 0).
    # ..#..   Every siding that touches the bottom one is given back, and only (1, 1) stays.
    # #...P   (With the bottom one as the main component only (3, 3) would stay.)
    # .....
    grid = CellGrid(5, 5, frozenset({(1, 4), (2, 2), (3, 0)}), ((0, 3), (3, 4)))
    assert grid_sidings(grid, "legacy") == [(1, 1)]

@pytest.mark.parametrize("mode", GRID_MODES)
def test_two_equal_rooms_without_a_door_keep_no_siding(mode):
    # a wall on column 4 cuts the grid in two equal rooms with 1 port each; a round gives the sidings of one room back,
    # which makes that room the bigger one, so the next round gives back the sidings of the other room
    grid = CellGrid(5, 9, frozenset((r, 4) for r in range(5)), ((0, 0), (0, 8)))
    assert grid_sidings(grid, mode) == []

@pytest.mark.parametrize("mode", GRID_MODES)
def test_a_siding_with_no_corridor_neighbour_is_dropped(mode):
    # (1, 1) has a wall on all four sides; the rounds leave it alone and only the last step drops it
    grid = CellGrid(3, 3, frozenset({(0, 1), (1, 0), (1, 2), (2, 1)}), ((0, 0),))
    assert grid_candidates(grid) == [(1, 1)]
    assert grid_sidings(grid, mode) == []
