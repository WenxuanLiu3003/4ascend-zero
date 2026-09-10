import numpy as np
import pytest

from src.core.board import Board
from src.core.engine import Engine
from src.core.rules import RulesConfig
from src.core.state import GameState
from src.core.types import Move, Player


def test_default_rule_is_relay():
    assert RulesConfig().relay_cancellation is True


@pytest.mark.parametrize("defender", [Player.BLACK, Player.WHITE])
@pytest.mark.parametrize("relay", [False, True])
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
    defender, relay, atk_power, def_power, attacker_damage, defender_damage
):
    if relay and attacker_damage == 3:
        attacker_damage = 2
    if relay and defender_damage == 3:
        defender_damage = 2
    state = GameState(cfg=RulesConfig(relay_cancellation=relay), board=Board(9))
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


def test_relay_damage_matches_sorted_power_prefixes():
    # Continuous cancellation spends the opposing total against a sorted
    # prefix. Count the cells that still have power after that expenditure.
    from itertools import combinations_with_replacement

    powers = [p for n in range(5) for p in combinations_with_replacement((1, 2, 3), n)]
    engine = Engine()
    for atk in powers:
        for defense in powers:
            state = GameState(cfg=RulesConfig(relay_cancellation=True), board=Board(9))
            state.hp[:] = 10
            for row, values in enumerate((atk, defense)):
                for col, value in enumerate(values):
                    state.board.plants[row, col] = value - 1
            engine._resolve_attack_defense(
                state, Player.WHITE, Move(1, 0),
                {(0, c) for c in range(len(atk))},
                {(1, c) for c in range(len(defense))},
            )
            atk_damage = sum(np.cumsum(sorted(defense, reverse=True)) > sum(atk)) if sum(defense) > sum(atk) else 0
            def_damage = sum(np.cumsum(sorted(atk, reverse=True)) > sum(defense)) if sum(atk) > sum(defense) else 0
            assert state.hp.tolist() == [10 - atk_damage, 10 - def_damage], (atk, defense)
