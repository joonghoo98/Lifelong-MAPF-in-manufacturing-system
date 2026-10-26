"""The rest-cell ledger of m6 (design §5.8): at most one vehicle per rest cell, one cell per vehicle.

A bundle may swap cells between its vehicles, so its new homes move all at once (move_all); SidingLedger moves one
vehicle at a time and would refuse a swap halfway.
"""
from collections.abc import Iterable, Mapping


class RestLedger:
    """Who rests on which rest cell."""

    def __init__(self, cells: Iterable[str]) -> None:
        self._cells = frozenset(str(c) for c in cells)
        self._owner: dict[str, int] = {}
        self._home: dict[int, str] = {}

    def assign(self, vehicle: int, cell: str) -> None:
        """Give cell to vehicle, which lets its old one go. ValueError if it is no rest cell or someone else's."""
        if cell not in self._cells:
            raise ValueError(f"{cell} is not a rest cell")
        cur = self._owner.get(cell)
        if cur is not None and cur != vehicle:
            raise ValueError(f"rest cell {cell} already belongs to vehicle {cur}")
        old = self._home.get(vehicle)
        if old is not None and old != cell:
            del self._owner[old]
        self._owner[cell] = vehicle
        self._home[vehicle] = cell

    def move_all(self, homes: Mapping[int, str]) -> None:
        """Give each vehicle in homes its cell at once. ValueError and no change if a cell is no rest cell, is wanted
        twice, or stays with a vehicle that is not in homes."""
        wanted = sorted(homes.values())
        if any(c not in self._cells for c in wanted):
            raise ValueError(f"not all of {wanted} are rest cells")
        if len(set(wanted)) < len(wanted):
            raise ValueError(f"a rest cell is wanted twice: {dict(homes)}")
        for vehicle, cell in sorted(homes.items()):
            cur = self._owner.get(cell)
            if cur is not None and cur not in homes:
                raise ValueError(f"rest cell {cell} stays with vehicle {cur}")
        for vehicle in sorted(homes):
            old = self._home.pop(vehicle, None)
            if old is not None:
                del self._owner[old]
        for vehicle, cell in sorted(homes.items()):
            self._owner[cell] = vehicle
            self._home[vehicle] = cell

    def home(self, vehicle: int) -> str | None:
        return self._home.get(vehicle)

    def owner(self, cell: str) -> int | None:
        return self._owner.get(cell)

    def snapshot(self) -> dict[int, str]:
        """vehicle -> its rest cell, in vehicle order."""
        return dict(sorted(self._home.items()))
