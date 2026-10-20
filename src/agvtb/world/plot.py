"""A static picture of a World: its segments, nodes and ports."""
from pathlib import Path

from matplotlib import patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from agvtb.world.model import World


def plot_world(world: World, out_path: Path, title: str | None = None) -> Path:
    """Save a PNG of the world at out_path, making its folder if needed, and return out_path.

    Segments are grey lines, nodes small dots, ports red markers labelled with their node id. Both
    axes have the same scale, in metres, with y up as in the map. The figure is drawn by Agg
    through the Figure class alone, so no window opens, the pyplot backend is left alone and no
    figure stays open afterwards.
    """
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=150, bbox_inches="tight")
    return out_path


def _split(points: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    """The x values and the y values of a list of (x, y) points."""
    return [x for x, _ in points], [y for _, y in points]
