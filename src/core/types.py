from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, List




class Player(Enum):
    BLACK = 0 # 先手
    WHITE = 1 # 后手


    def other(self) -> "Player":
        return Player.BLACK if self is Player.WHITE else Player.WHITE




class Phase(Enum):
    NORMAL = auto() # 普通落子阶段
    ATTACK_DEFENSE = auto() # 进入“攻防状态”的结算阶段




@dataclass(frozen=True)
class Move:
    r: int
    c: int
    # 如需扩展（技能、种植等），可在此加入字段或改为 Union 类型


    def to_tuple(self) -> Tuple[int, int]:
        return (self.r, self.c)