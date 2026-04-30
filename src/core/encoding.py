from __future__ import annotations
import numpy as np
from typing import List

from .state import GameState
from .types import Player
from .types import Phase

class AlphaZeroStateEncoder:
    """
    Encode GameState into a fixed set of planes for neural network input. 
    The encoding is designed to be simple and informative, with a total of 25 planes:
    """

    def __init__(self, last_k: int = 8):
        # last_k: historical planes to include, do not change
        self.last_k = last_k

    @staticmethod
    def _adjacent_run_lengths(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
        """
        返回每个位置沿 (dr, dc) 方向、从相邻格开始的连续同色长度（不含自身）。
        使用单次扫描 DP，复杂度 O(HW)。
        """
        H, W = mask.shape
        out = np.zeros((H, W), dtype=np.int16)

        r_iter = range(H - 1, -1, -1) if dr > 0 else range(H)
        c_iter = range(W - 1, -1, -1) if dc > 0 else range(W)

        for r in r_iter:
            nr = r + dr
            if nr < 0 or nr >= H:
                continue
            for c in c_iter:
                nc = c + dc
                if nc < 0 or nc >= W:
                    continue
                if mask[nr, nc]:
                    out[r, c] = out[nr, nc] + 1
        return out

    @classmethod
    def _ge4_axis_sum_plane(cls, grid: np.ndarray, pid: int) -> np.ndarray:
        """
        For any given position, assuming a stone of player pid is placed there, count the sum of lengths of potential >=4-in-a-row in four axes.
        The center point is only counted once: if multiple axes are satisfied, do not double count the center point.
        By design: if the position is already occupied by pid, we can directly set it to 0 for acceleration/denoising.
        """
        mask = (grid == pid)

        up = cls._adjacent_run_lengths(mask, -1, 0)
        down = cls._adjacent_run_lengths(mask, 1, 0)
        left = cls._adjacent_run_lengths(mask, 0, -1)
        right = cls._adjacent_run_lengths(mask, 0, 1)
        up_left = cls._adjacent_run_lengths(mask, -1, -1)
        down_right = cls._adjacent_run_lengths(mask, 1, 1)
        up_right = cls._adjacent_run_lengths(mask, -1, 1)
        down_left = cls._adjacent_run_lengths(mask, 1, -1)

        v_nb = up + down
        h_nb = left + right
        d45_nb = up_right + down_left
        d135_nb = up_left + down_right

        nb_stack = np.stack([v_nb, h_nb, d45_nb, d135_nb], axis=0)  
        valid = (nb_stack + 1) >= 4                                   

        score = np.where(valid, nb_stack, 0).sum(axis=0).astype(np.int16)
        score += valid.any(axis=0).astype(np.int16)

        score = np.where(mask, 0, score)
        return score.astype(np.float32)

    def encode(self, s: GameState, as_player: Player) -> np.ndarray:
        H, W = s.board.grid.shape
        planes: List[np.ndarray] = []

        grid = s.board.grid
        me_id = 1 if as_player is Player.BLACK else 2
        opp_id = 2 if as_player is Player.BLACK else 1

        # plane 0,1: the current stone distribution for both sides (me and opponent)
        me = (grid == me_id).astype(np.float32)
        opp = (grid == opp_id).astype(np.float32)
        planes.extend([me, opp])

        # plane 2: attack chain mask (only for ascend phase, otherwise all zeros)
        attack_mask = (
            s.attack_chain_mask.astype(np.float32)
            if s.attack_chain_mask is not None else np.zeros((H, W), dtype=np.float32)
        )
        planes.append(attack_mask)

        # plane 3: plant count
        plants_count = s.board.plants.astype(np.float32)
        planes.extend([plants_count])

        # plane 4,5: HP status for both sides
        hp_max = max(1, s.cfg.hp_max)
        me_idx = 0 if as_player.name == "BLACK" else 1
        opp_idx = 1 - me_idx
        me_hp = np.full((H, W), float(s.hp[me_idx]), dtype=np.float32)
        opp_hp = np.full((H, W), float(s.hp[opp_idx]), dtype=np.float32)
        planes.extend([me_hp, opp_hp])

        # plane 6: whether it's currently ascend phase
        is_ad = np.full((H, W), 1.0 if s.phase.name == "ATTACK_DEFENSE" else 0.0, dtype=np.float32)
        # plane 7: whether it's currently the turn of the first player
        my_turn = np.full((H, W), 1.0 if s.to_play is Player.BLACK else 0.0, dtype=np.float32)
        planes.extend([is_ad, my_turn])

        # plane 8: grow count parity 
        if s.phase is Phase.ATTACK_DEFENSE:
            grow_parity = np.full((H, W), 1, dtype=np.float32)
        else:
            grow_parity = np.full((H, W), float(s.grow_count % 2), dtype=np.float32)
        planes.append(grow_parity)

        # plane 9,10: potential >=4-in-a-row lengths for me and opponent at each position (if I/they place a stone there)
        me_ge4_axis_sum = self._ge4_axis_sum_plane(grid, me_id)
        opp_ge4_axis_sum = self._ge4_axis_sum_plane(grid, opp_id)
        planes.extend([me_ge4_axis_sum, opp_ge4_axis_sum])


        # planes 11-24: historical stone distributions for both sides in the last 7 moves (if available), from old to new. If not enough history, fill with zeros.
        for j in range(1, self.last_k):
            if j > len(s.last_moves):
                planes.extend([np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)])
                continue
            past_grid = s.last_moves[-j]
            if past_grid is None:
                continue
            move_index = s.turn - j 
            p = 1 if move_index % 2 == 0 else 2
            opp_p = 2 if p == 1 else 1
            me = (past_grid == p).astype(np.float32)
            opp = (past_grid  == opp_p).astype(np.float32)
            planes.extend([me, opp])

        out = np.stack(planes, axis=0)  # [C,H,W]
        assert out.shape[0] == 25, f"expect 25 planes, got {out.shape[0]}"
        return out

    @property
    def num_planes(self) -> int:
        return 25
