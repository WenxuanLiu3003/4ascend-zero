# ──────────────────────────────────────────────────────────────────────────────
# File: src/core/board.py
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Optional, List, Tuple


@dataclass
class Board:
    size: int
    # 0=空, 1=Black, 2=White
    grid: np.ndarray = field(init=False)
    # 植物数量：≥0 的整数（允许一个格子≥2 个魔力植物）
    plants: np.ndarray = field(init=False)

    def __post_init__(self):
        self.grid = np.zeros((self.size, self.size), dtype=np.uint8)
        # 使用无符号整型计数器，便于累加/扣减
        self.plants = np.zeros((self.size, self.size), dtype=np.uint16)

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def is_empty(self, r: int, c: int) -> bool:
        return self.grid[r, c] == 0

    def place_stone(self, r: int, c: int, player: int) -> None:
        # player: 1 for Black, 2 for White
        assert self.in_bounds(r, c), "out of bounds"
        assert self.is_empty(r, c), "cell not empty"
        self.grid[r, c] = player

    # —— 植物相关 ——
    def add_plants(self, r: int, c: int, count: int = 1) -> None:
        assert self.in_bounds(r, c)
        assert count >= 0
        self.plants[r, c] = np.uint16(self.plants[r, c] + count)

    def remove_plants(self, r: int, c: int, count: int | None = None) -> None:
        assert self.in_bounds(r, c)
        if count is None:
            self.plants[r, c] = np.uint16(0)
        else:
            if count >= self.plants[r, c]:
                self.plants[r, c] = np.uint16(0)
            else:
                self.plants[r, c] = np.uint16(self.plants[r, c] - count)

    def remove_stones(self, coords: List[Tuple[int, int]]) -> None:
        for r, c in coords:
            if self.in_bounds(r, c):
                self.grid[r, c] = 0

    def clone(self) -> "Board":
        b = Board(self.size)
        b.grid = self.grid.copy()
        b.plants = self.plants.copy()
        return b

