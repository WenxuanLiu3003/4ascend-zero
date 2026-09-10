import numpy as np
import pytest

from src.core.board import Board
from src.core.engine import Engine
from src.core.rules import RulesConfig
from src.core.state import GameState
from src.core.types import Move, Player


@pytest.mark.parametrize("defender", [Player.BLACK, Player.WHITE])
@pytest.mark.parametrize(
    "atk_power, def_power, attacker_damage, defender_damage",
    [
        ([3, 3, 3, 3], [2, 2, 2, 2], 0, 2),
        ([3, 1, 1, 1], [2, 2, 2, 2, 1], 3, 0),
        ([2, 2, 2, 2], [3, 3, 3, 3], 2, 0),
        ([2, 2, 2, 2, 1], [3, 1, 1, 1], 0, 3),
        ([3], [3], 0, 0),
        ([3, 1], [], 0, 2),
        ([], [3, 1], 2, 0),
        ([], [], 0, 0),
    ],
)
def test_attack_defense_damage(
    defender, atk_power, def_power, attacker_damage, defender_damage
):
    state = GameState(cfg=RulesConfig(), board=Board(9))
    state.hp[:] = 10
    atk_cells = {(0, c) for c in range(len(atk_power))}
    def_cells = {(1, c) for c in range(len(def_power))}
    for row, powers in enumerate((atk_power, def_power)):
        for col, power in enumerate(powers):
            state.board.plants[row, col] = power - 1

    Engine()._resolve_attack_defense(
        state, defender, Move(1, 0), atk_cells, def_cells
    )

    expected_hp = [10, 10]
    expected_hp[defender.other().value] -= attacker_damage
    expected_hp[defender.value] -= defender_damage
    np.testing.assert_array_equal(state.hp, expected_hp)
