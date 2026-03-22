from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, List




class Player(Enum):
    BLACK = 0 # the first to place the stone
    WHITE = 1


    def other(self) -> "Player":
        return Player.BLACK if self is Player.WHITE else Player.WHITE




class Phase(Enum):
    NORMAL = auto()
    ATTACK_DEFENSE = auto()




@dataclass(frozen=True)
class Move:
    r: int
    c: int


    def to_tuple(self) -> Tuple[int, int]:
        return (self.r, self.c)