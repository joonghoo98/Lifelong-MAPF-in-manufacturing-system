"""Job demand: a from-to table of rates and one exponential arrival stream per source port (design §7, D6).

The table maps (source port, destination port) to a rate in jobs/s whose total is one job a minute (1/60 jobs/s);
the run's rate_per_min scales it. Each source port draws its gaps with expovariate(sum of its row x rate_per_min)
and its destination in proportion to the row. One random.Random drives every port, and an arrival draws the
destination first, then the next gap (DAIM mcs/mcs_engine.py:1720-1749, 2634-2686, research §E). Ports are taken
in sorted order (DAIM takes a string-hash order).
"""
import csv
import random
from collections.abc import Iterable, Mapping
from pathlib import Path

PER_MINUTE = 1.0 / 60.0  # one job a minute, in jobs/s


def read_fromto(path: str | Path, ports: Iterable[str]) -> dict[tuple[str, str], float]:
    """A from-to CSV: first row the destination ports (first cell empty), first column the source ports, rates in
    jobs/s. Rows and columns that are no port, src == dst, and empty, zero or negative cells are left out."""
    port_set = set(ports)
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}
    dsts = [c.strip() for c in rows[0][1:]]
    out: dict[tuple[str, str], float] = {}
    for row in rows[1:]:
        if not row:
            continue
        src = row[0].strip()
        if src not in port_set:
            continue
        for dst, cell in zip(dsts, row[1:]):
            cell = cell.strip()
            if dst not in port_set or dst == src or cell in ("", "0"):
                continue
            lam = float(cell)
            if lam > 0:
                out[(src, dst)] = lam
    return out


def uniform_fromto(ports: Iterable[str]) -> dict[tuple[str, str], float]:
    """Every ordered pair of distinct ports at one rate, one job a minute in all."""
    p = sorted(set(ports))
    pairs = [(a, b) for a in p for b in p if a != b]
    return {pair: PER_MINUTE / len(pairs) for pair in pairs}


def normalize(fromto: Mapping[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    """The table scaled to one job a minute in all. ValueError if it has no positive rate."""
    total = sum(fromto.values())
    if not total > 0:
        raise ValueError(f"a from-to table needs a positive rate, its total is {total}")
    return {pair: lam * PER_MINUTE / total for pair, lam in fromto.items()}


class Demand:
    def __init__(self, fromto: Mapping[tuple[str, str], float], rate_per_min: float, rng: random.Random):
        if not rate_per_min >= 0:
            raise ValueError(f"rate_per_min must be >= 0, got {rate_per_min}")
        self.rng = rng
        self._rows: dict[str, dict[str, float]] = {}
        for (src, dst), lam in sorted(fromto.items()):
            self._rows.setdefault(src, {})[dst] = lam
        self._rate = {src: sum(row.values()) * rate_per_min for src, row in self._rows.items()}

    def rates(self) -> dict[str, float]:
        """Source port -> its arrival rate in jobs/s (ports with none left out)."""
        return {src: r for src, r in sorted(self._rate.items()) if r > 0}

    def first_arrivals(self) -> list[tuple[float, str]]:
        return [(self.rng.expovariate(r), src) for src, r in self.rates().items()]

    def on_arrival(self, t: float, src: str) -> tuple[str, float]:
        """The new job's destination, then the time of the port's next arrival."""
        row = self._rows[src]
        dsts = sorted(row)
        dst = self.rng.choices(dsts, weights=[row[d] for d in dsts])[0]
        return dst, t + self.rng.expovariate(self._rate[src])
