import pytest

from agvtb.planning.rest_ledger import RestLedger


def ledger():
    led = RestLedger(["a", "b", "c", "d"])
    for v, c in enumerate("abc"):
        led.assign(v, c)
    return led


def test_move_all_lets_vehicles_swap_cells():
    led = ledger()
    led.move_all({0: "b", 1: "d", 2: "a"})
    assert led.snapshot() == {0: "b", 1: "d", 2: "a"} and led.owner("c") is None and led.home(1) == "d"


@pytest.mark.parametrize("homes", [{0: "b"}, {0: "d", 1: "d"}, {0: "x"}])
def test_move_all_refuses_and_changes_nothing(homes):
    led = ledger()
    with pytest.raises(ValueError):
        led.move_all(homes)
    assert led.snapshot() == {0: "a", 1: "b", 2: "c"}


def test_assign_refuses_a_cell_someone_owns():
    with pytest.raises(ValueError):
        ledger().assign(3, "a")
