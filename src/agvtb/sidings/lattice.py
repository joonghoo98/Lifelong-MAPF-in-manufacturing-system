"""A lattice map as a cell grid: the step from a World to the CellGrid the grid rule of the sidings reads.

DAIM's cell model is a rows x cols block of cells. Every free cell is a node, and every two free cells
that share an edge are joined by one segment each way (mapgen/grid_layout_to_xmsmap.py). lattice_grid
finds that block in a World, checks that the map keeps to the model, and numbers the cells. A map that
does not (KAIROS_AMR) gets a LatticeError, not a wrong grid.
"""
import collections
from dataclasses import dataclass

from agvtb.sidings.grid_rule import Cell, CellGrid, grid_sidings
from agvtb.world.model import Segment, World

GRID_SHIFTS = ((0, 0), (0, 1), (1, 0), (1, 1))  # (dr, dc): blocked rows below and columns left of the lattice
_LENGTH_DIGITS = 6  # segment lengths are counted rounded to 1e-6 m
_LATTICE_TOLERANCE_M = 1e-6  # absolute slack for matching a node to its lattice point (rounded coordinates)


class LatticeError(ValueError):
    """The map is not a lattice map the grid rule can read."""


@dataclass(frozen=True)
class LatticeGrid:
    """A World's lattice as a CellGrid, with the maps between its nodes and cells."""

    grid: CellGrid
    cell_of: dict[str, Cell]  # lattice node -> its cell; nodes off the lattice are not in it
    node_at: dict[Cell, str]  # cell -> its lattice node (the inverse of cell_of)
    access: dict[str, str]  # off-lattice port -> the lattice node it is linked to (its access node)
    pitch_m: float
    shift: tuple[int, int]


def lattice_grid(world: World, shift: tuple[int, int] = (0, 0)) -> LatticeGrid:
    """The lattice of world as a CellGrid; shift = (dr, dc) from GRID_SHIFTS moves the origin.

    The pitch is the actual length of the first segment, in map file order, of the most common length
    (the shorter one on a tie), and the lattice is anchored at that segment's start node. A node within
    1e-6 m of a lattice point, in x and in y, is a lattice node and becomes a cell. Cell (0, 0) is the lower
    left of the lattice nodes, moved by shift: dr blocked rows below and dc blocked columns left. With shift
    (0, 0) that is DAIM's own numbering when DAIM's grid has no blocked row below and no blocked column left of
    the lattice nodes. When it has some, take the member of GRID_SHIFTS equal to (the blocked rows below, the
    blocked columns left) mod 2: the grid rule reads a cell number only through the parity of its row and of
    its column, so that shift picks the sidings DAIM's numbering does, and load_daim_grid is not needed for it.
    Any other node must be a port linked to exactly one node, a lattice node (its access node), and stands
    in that node's cell. The ports of the grid follow world.port_nodes, each cell once.

    Raises ValueError if shift is not in GRID_SHIFTS. Raises LatticeError, naming the world and the nodes at
    fault, if two nodes share a lattice point; a segment between lattice nodes does not join neighbouring
    cells; neighbouring lattice nodes are not linked both ways; or a node off the lattice is no port or is
    not linked to exactly one lattice node. A node off the lattice that is no port is reported first, with
    the count of the nodes off the lattice (the map is no lattice); a port off it is looked at after that.
    """
    if shift not in GRID_SHIFTS:
        raise ValueError(f"grid shift must be one of {', '.join(map(str, GRID_SHIFTS))}, got {shift!r}")
    shift = GRID_SHIFTS[GRID_SHIFTS.index(shift)]  # the plain ints, whatever equal value came in
    dr, dc = shift
    if not world.segments:
        raise LatticeError(f"{world.name}: the map has no segments")
    ref = _reference_segment(world)
    pitch = ref.length_m
    at_point, off_lattice = _place_nodes(world, ref.u, pitch)
    row0, col0 = min(row for row, _ in at_point), min(col for _, col in at_point)
    cell_of = {node: (row - row0 + dr, col - col0 + dc) for (row, col), node in at_point.items()}
    node_at = {cell: node for node, cell in cell_of.items()}
    _check_cell_model(world, cell_of, node_at)
    access = _access_nodes(world, cell_of, off_lattice, ref.u, pitch)

    rows, cols = max(row for row, _ in node_at) + 1, max(col for _, col in node_at) + 1
    blocked = frozenset((r, c) for r in range(rows) for c in range(cols) if (r, c) not in node_at)
    # an off-lattice port stands at its access node; dict.fromkeys keeps the order and drops repeated cells
    ports = tuple(dict.fromkeys(cell_of[access.get(port, port)] for port in world.port_nodes))
    return LatticeGrid(CellGrid(rows, cols, blocked, ports), cell_of, node_at, access, pitch, shift)


def grid_sidings_of_world(world: World, mode: str = "bridge", shift: tuple[int, int] = (0, 0)) -> list[str]:
    """The sidings the grid rule picks on the lattice of world, as node ids in cell order.

    That is the order of DAIM's info.sidings. Raises what lattice_grid and grid_sidings raise.
    """
    lattice = lattice_grid(world, shift)
    return [lattice.node_at[cell] for cell in grid_sidings(lattice.grid, mode)]


def _reference_segment(world: World) -> Segment:
    """The first segment, in map file order, of the most common length; the shorter length wins a tie.

    Lengths are counted rounded to 1e-6 m. M1's grid_pitch, the shortest segment, is not used: it picks
    the 0.478 m port links of songdo_grid_bi.
    """
    counts = collections.Counter(round(seg.length_m, _LENGTH_DIGITS) for seg in world.segments.values())
    common = max(counts, key=lambda length: (counts[length], -length))
    return next(seg for seg in world.segments.values() if round(seg.length_m, _LENGTH_DIGITS) == common)


def _place_nodes(
    world: World, anchor: str, pitch: float
) -> tuple[dict[tuple[int, int], str], list[str]]:
    """Split the nodes into those on the lattice of spacing pitch anchored at node anchor, and the rest.

    Returns {(row, col) counted from the anchor: node} for the lattice nodes and the list of the other
    nodes, both in node order. Raises LatticeError if two nodes sit at one lattice point.
    """
    ax, ay = world.nodes[anchor]
    at_point: dict[tuple[int, int], str] = {}
    off_lattice: list[str] = []
    for node, (x, y) in world.nodes.items():
        col, row = round((x - ax) / pitch), round((y - ay) / pitch)  # the nearest lattice point
        if max(abs(x - (ax + col * pitch)), abs(y - (ay + row * pitch))) > _LATTICE_TOLERANCE_M:
            off_lattice.append(node)
        elif (row, col) in at_point:
            raise LatticeError(
                f"{world.name}: nodes {at_point[(row, col)]} and {node} are at the same lattice point"
            )
        else:
            at_point[(row, col)] = node
    return at_point, off_lattice


def _check_cell_model(world: World, cell_of: dict[str, Cell], node_at: dict[Cell, str]) -> None:
    """Raise LatticeError unless the lattice nodes keep to DAIM's cell model.

    A segment between two lattice nodes joins cells one step apart along an axis, and two lattice nodes
    in neighbouring cells are linked both ways (grid_layout_to_xmsmap.py writes a pair for each).
    """
    for u, v in world.segments:
        if u in cell_of and v in cell_of:
            (r0, c0), (r1, c1) = cell_of[u], cell_of[v]
            if abs(r1 - r0) + abs(c1 - c0) != 1:
                raise LatticeError(f"{world.name}: segment {u} -> {v} does not join neighbouring cells")
    for node, (row, col) in cell_of.items():
        for other in (node_at.get((row + 1, col)), node_at.get((row, col + 1))):  # each pair is seen once
            if other is None:
                continue
            for start, end in ((node, other), (other, node)):
                if (start, end) not in world.segments:
                    raise LatticeError(
                        f"{world.name}: nodes {node} and {other} are in neighbouring cells "
                        f"but there is no segment {start} -> {end}"
                    )


def _access_nodes(
    world: World, cell_of: dict[str, Cell], off_lattice: list[str], anchor: str, pitch: float
) -> dict[str, str]:
    """Map each node off the lattice to its access node, the one lattice node it is linked to.

    Only a port may be off the lattice, so the nodes off it that are no port are looked at first. If there is
    one, LatticeError counts the nodes off the lattice (of spacing pitch, anchored at node anchor) and those that
    are no port, and names the first of the latter in node order: a map like KAIROS_AMR is no lattice as a
    whole, whatever its ports do. Then the ports off the lattice: the nodes joined to one by a segment, in
    either direction, are the ones it is linked to. Raises LatticeError if a port is not linked to exactly one
    node, or that node is off the lattice too.
    """
    ports = set(world.port_nodes)
    no_port = [node for node in off_lattice if node not in ports]
    if no_port:
        raise LatticeError(
            f"{world.name}: {len(off_lattice)} of {len(world.nodes)} nodes are off the {pitch:g} m lattice of {anchor} "
            f"({len(no_port)} are no port), first {no_port[0]}: only a port may be off the lattice"
        )
    linked: dict[str, set[str]] = {node: set() for node in off_lattice}
    for u, v in world.segments:
        if u in linked:
            linked[u].add(v)
        if v in linked:
            linked[v].add(u)
    access: dict[str, str] = {}
    for node in off_lattice:
        if len(linked[node]) != 1:
            raise LatticeError(
                f"{world.name}: port {node} is off the lattice and must be linked to exactly one node, "
                f"but is linked to {len(linked[node])} ({', '.join(sorted(linked[node]))})"
            )
        (neighbour,) = linked[node]
        if neighbour not in cell_of:
            raise LatticeError(
                f"{world.name}: port {node} is off the lattice and linked to {neighbour}, which is off it too"
            )
        access[node] = neighbour
    return access
