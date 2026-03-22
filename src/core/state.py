from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from .types import Player, Phase, Move
from .board import Board
from .rules import RulesConfig


@dataclass
class GameState:
    cfg: RulesConfig
    board: Board
    to_play: Player = Player.BLACK
    turn: int = 0
    phase: Phase = Phase.NORMAL

    # historical last k moves for plane encoding
    last_moves: List[Optional[Move]] = field(default_factory=list)
    last_k_for_planes: int = 4

    # HP status
    hp: np.ndarray = field(default_factory=lambda: np.array([RulesConfig.hp_max, RulesConfig.hp_max], dtype=np.int32))

    # recoding the attackers attacking chain
    attack_chain_mask: Optional[np.ndarray] = None  # 攻方“可被无效化”的连子位置掩码（H×W，bool）

    # the last move
    last_move: Optional[Move] = None

    # the state variables used for refreshing the plant
    over_fill: bool = False
    just_unascend: bool = False
    grow_count: int = 11
    unascend_charge: int = 25
    Aunascend_charge_fast: np.ndarray = field(default_factory=lambda: np.array([9, 9], dtype=np.int32))

    def copy(self) -> "GameState":
        return GameState(
            cfg=self.cfg,
            board=self.board.clone(),
            to_play=self.to_play,
            turn=self.turn,
            phase=self.phase,
            last_moves=list(self.last_moves),
            last_k_for_planes=self.last_k_for_planes,
            hp=self.hp.copy(),
            attack_chain_mask=None if self.attack_chain_mask is None else self.attack_chain_mask.copy(),
            last_move=self.last_move,
            over_fill=self.over_fill,
            just_unascend=self.just_unascend,
            grow_count=self.grow_count,
            unascend_charge=self.unascend_charge,
            Aunascend_charge_fast=self.Aunascend_charge_fast.copy(),
        )
    
    # whether the game has ended
    def is_terminal(self) -> bool:
        # 1) someone's HP is 0 or below
        if (self.hp <= 0).any():
            return True
        # 2) reaches the maximum turn limit
        if self.turn >= self.cfg.max_turns:
            return True
        # 3) board  is fully filled
        if self.phase == Phase.NORMAL and not np.any(self.board.grid == 0):
            return True
        return False

    def legal_moves(self) -> List[Move]:
        empties = np.argwhere(self.board.grid == 0)
        return [Move(int(r), int(c)) for r, c in empties]
