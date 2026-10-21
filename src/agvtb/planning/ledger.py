"""The siding ledger: at most one vehicle per siding, exactly one siding per vehicle (DAIM :54-96)."""
from collections.abc import Iterable


class SidingLedger:
    """Who owns which siding. A vehicle may end a task on a siding nobody else owns, or on its own."""

    def __init__(self, sidings: Iterable[str]):
        self._sidings = sorted({str(s) for s in sidings})
        self._set = set(self._sidings)
        self._owner: dict[str, int] = {}
        self._home: dict[int, str] = {}

    @property
    def sidings(self) -> list[str]:
        return list(self._sidings)

    def assign(self, vehicle: int, siding: str) -> None:
        """Give siding to vehicle, which lets its old one go. ValueError if it is no siding or someone else's."""
        siding = str(siding)
        if siding not in self._set:
            raise ValueError(f"{siding} is not a siding")
        cur = self._owner.get(siding)
        if cur is not None and cur != vehicle:
            raise ValueError(f"siding {siding} already belongs to vehicle {cur}")
        old = self._home.get(vehicle)
        if old is not None and old != siding:
            self._owner.pop(old, None)
        self._owner[siding] = vehicle
        self._home[vehicle] = siding

    def transfer(self, vehicle: int, siding: str) -> None:
        """At a commit, the tail's siding becomes the vehicle's (the checks of assign)."""
        self.assign(vehicle, siding)

    def home(self, vehicle: int) -> str | None:
        return self._home.get(vehicle)

    def owner(self, siding: str) -> int | None:
        return self._owner.get(str(siding))

    def candidates(self, vehicle: int) -> list[str]:
        """The sidings vehicle may end on: those nobody owns and its own (sorted)."""
        return [s for s in self._sidings if self._owner.get(s, vehicle) == vehicle]

    def snapshot(self) -> dict[int, str]:
        """vehicle -> its siding, in vehicle order."""
        return dict(sorted(self._home.items()))
