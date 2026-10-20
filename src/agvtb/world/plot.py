"""A static picture of a World: its segments, nodes and ports, and the sidings and excluded candidates on it."""
from collections.abc import Iterable, Mapping
from pathlib import Path

from matplotlib import patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from agvtb.world.model import World

# Marker area [pt^2] of a siding square: small, so that the squares of nodes 1 m apart (songdo_grid_bi) do not touch.
_SIDING_AREA = 14
# Marker area of the black cross of each reason a candidate was excluded for (the reasons of condition M, see
# agvtb.sidings.condition_m.EXCLUSION_REASONS). The crosses cannot differ in colour, so they differ in size, and the
# same reason has the same size in every figure.
_CROSS_AREA = {"pocket": 30, "port_access": 50, "port": 70, "not_in_graph": 90}
_OTHER_CROSS_AREA = 110  # any other reason


def plot_world(
    world: World,
    out_path: Path,
    title: str | None = None,
    *,
    sidings: Iterable[str] = (),
    excluded: Mapping[str, str] | None = None,
) -> Path:
    """Save a PNG of the world at out_path, making its folder if needed, and return out_path.

    Segments are grey lines, nodes small dots, ports red markers labelled with their node id. Both
    axes have the same scale, in metres, with y up as in the map. The figure is drawn by Agg
    through the Figure class alone, so no window opens, the pyplot backend is left alone and no
    figure stays open afterwards.

    sidings (node ids) are marked by blue squares and the candidates in excluded (node id -> reason) by black
    crosses, one size for each reason. Both lie under the ports and their labels. When there are crosses, a
    legend at the right of the map counts the crosses of each reason and the sidings; without crosses there is
    no legend, and with nothing marked the figure is the one made without these arguments. Raises ValueError
    if a siding is no node of the world. An excluded candidate that is no node (reason not_in_graph) has no
    place on the map and is left out.
    """
    marked = sorted(set(sidings))
    for node in marked:
        if node not in world.nodes:
            raise ValueError(f"{world.name}: siding {node!r} is no node of the world")
    crosses: dict[str, list[tuple[float, float]]] = {}  # reason -> the places of its candidates
    for node, reason in sorted((excluded or {}).items()):
        if node in world.nodes:
            crosses.setdefault(reason, []).append(world.nodes[node])

    fig = Figure(figsize=(10, 10))
    FigureCanvasAgg(fig)  # puts the Agg canvas on fig
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    ax.margins(0.15)  # room at the edges for the port labels
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if title is not None:
        ax.set_title(title)

    # one LineCollection, not one artist per segment: songdo_grid_bi has 4,838
    ax.add_collection(LineCollection(
        [(world.nodes[u], world.nodes[v]) for u, v in world.segments], colors="0.7", linewidths=0.8, zorder=1,
    ))
    ax.scatter(*_split(list(world.nodes.values())), s=4, color="0.3", zorder=2)
    port_xy = [world.nodes[node] for node in world.port_nodes]
    for node, xy in zip(world.port_nodes, port_xy):
        # Slanted, so the labels of ports a cell apart in a row or a column keep clear of each other;
        # the white halo keeps them readable over nodes and lines. Two ports on a rising diagonal
        # a cell apart can still hide part of a label.
        ax.annotate(
            node, xy, xytext=(3, 3), textcoords="offset points", fontsize=7, color="red", zorder=3,
            rotation=30, rotation_mode="anchor", ha="left", va="bottom",
            path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")],
        )
    ax.scatter(*_split(port_xy), s=40, color="red", zorder=4)  # above the labels: no marker is covered

    # over the node dots (2), under the port labels (3) and markers (4)
    if marked:
        ax.scatter(
            *_split([world.nodes[node] for node in marked]), s=_SIDING_AREA, marker="s", color="tab:blue",
            linewidths=0, zorder=2.5, label=f"siding ({len(marked)})",
        )
    for reason in sorted(crosses, key=lambda r: (_CROSS_AREA.get(r, _OTHER_CROSS_AREA), r)):
        ax.scatter(
            *_split(crosses[reason]), s=_CROSS_AREA.get(reason, _OTHER_CROSS_AREA), marker="x", color="black",
            linewidths=1.2, zorder=2.5, label=f"{reason} ({len(crosses[reason])})",
        )
    if crosses:  # the legend is for the crosses, and never empty: every cross group is labelled (an empty one warns)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=150, bbox_inches="tight")
    return out_path


def _split(points: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    """The x values and the y values of a list of (x, y) points."""
    return [x for x, _ in points], [y for _, y in points]
