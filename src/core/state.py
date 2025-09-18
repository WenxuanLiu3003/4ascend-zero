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

    # 历史落点（用于 AZ-lite 的“历史 k 步落点叠图”）
    last_moves: List[Optional[Move]] = field(default_factory=list)
    last_k_for_planes: int = 4

    # HP（示例：每回合或结算时可变化）
    hp: np.ndarray = field(default_factory=lambda: np.array([RulesConfig.hp_max, RulesConfig.hp_max], dtype=np.int32))

    # 攻防状态的临时数据（在 ATTACK_DEFENSE 时有效）
    attack_chain_mask: Optional[np.ndarray] = None  # 攻方“可被无效化”的连子位置掩码（H×W，bool）

    # 记录最近一次落点（冗余但方便）
    last_move: Optional[Move] = None

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
        )
    
    # 基础查询与终局
    def is_terminal(self) -> bool:
        # 1) 任一方 HP 归零
        if (self.hp <= 0).any():
            return True
        # 2) 达到最大步数
        if self.turn >= self.cfg.max_turns:
            return True
        # 3) 棋盘已满且不在攻防阶段
        if self.phase == Phase.NORMAL and not np.any(self.board.grid == 0):
            return True
        return False

    def legal_moves(self) -> List[Move]:
        # 普通落子：所有空位
        empties = np.argwhere(self.board.grid == 0)
        return [Move(int(r), int(c)) for r, c in empties]