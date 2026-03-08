from __future__ import annotations
import numpy as np
from typing import List

from .state import GameState
from .types import Player
from .types import Phase

class AlphaZeroStateEncoder:
    """
    将 GameState 编码为 [C,H,W] 的多通道张量，用于策略-价值网络输入。

    通道定义（总计 25 个；以 as_player 视角构建）：
    0: 我方棋（one-hot）
    1: 对方棋（one-hot）
    2: 攻方“可被无效化”的连子掩码（ascend才有）
    3: 植物“数量”通道（float，计数，0/1/2 ...）
    4: 我方 HP 归一化平铺（/hp_max）
    5: 对方 HP 归一化平铺（/hp_max）
    6: 是否处于ascend阶段（ascend）
    7: 当前是否轮到黑棋（my_turn）
    8: grow_count 奇偶（s.grow_count % 2）
    9: 若该点视为我方棋子，四轴向可形成的“>=4连”长度和
    10: 若该点视为对方棋子，四轴向可形成的“>=4连”长度和
    11 - 24: 历史7步的[当前棋子分布, 对手棋子分布]
    """

    def __init__(self, last_k: int = 8):
        # 虽然 last_k 可调，但当前通道布局固定为 4；若要更长历史，可同步修改下方拼接逻辑
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
        对每个位置假设落成 pid，统计四个轴向中可形成的 n>=4 连子长度和。
        中心点只计一次：多个轴向同时满足时，不重复累计中心点。
        按需求：若该位置已是 pid，则该位置直接记 0。
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

        # 每个轴向的“邻居连子数”（不含中心点）
        v_nb = up + down
        h_nb = left + right
        d45_nb = up_right + down_left
        d135_nb = up_left + down_right

        nb_stack = np.stack([v_nb, h_nb, d45_nb, d135_nb], axis=0)   # [4,H,W]
        valid = (nb_stack + 1) >= 4                                   # 该轴向是否可形成 >=4（含中心）

        # 先累计各轴向邻居数，再给“至少一个轴向有效”的格点补一次中心点，避免重复。
        score = np.where(valid, nb_stack, 0).sum(axis=0).astype(np.int16)
        score += valid.any(axis=0).astype(np.int16)

        # Note2: 若该位置本来就是 pid，则可直接设为 0 加速/降噪。
        score = np.where(mask, 0, score)
        return score.astype(np.float32)

    def encode(self, s: GameState, as_player: Player) -> np.ndarray:
        H, W = s.board.grid.shape
        planes: List[np.ndarray] = []

        grid = s.board.grid
        me_id = 1 if as_player is Player.BLACK else 2
        opp_id = 2 if as_player is Player.BLACK else 1

        # —— 基础棋面 ——
        me = (grid == me_id).astype(np.float32)
        opp = (grid == opp_id).astype(np.float32)
        # empty = (grid == 0).astype(np.float32)
        planes.extend([me, opp])

        # —— 攻方“可被无效化”掩码 ——
        attack_mask = (
            s.attack_chain_mask.astype(np.float32)
            if s.attack_chain_mask is not None else np.zeros((H, W), dtype=np.float32)
        )
        planes.append(attack_mask)

        # —— 植物数量 ——
        plants_count = s.board.plants.astype(np.float32)
        # plants_mask = (s.board.plants > 0).astype(np.float32)
        planes.extend([plants_count])

        # —— HP 归一化 ——
        hp_max = max(1, s.cfg.hp_max)
        me_idx = 0 if as_player.name == "BLACK" else 1
        opp_idx = 1 - me_idx
        me_hp = np.full((H, W), float(s.hp[me_idx]), dtype=np.float32)
        opp_hp = np.full((H, W), float(s.hp[opp_idx]), dtype=np.float32)
        planes.extend([me_hp, opp_hp])

        # —— 阶段与行动方 ——
        is_ad = np.full((H, W), 1.0 if s.phase.name == "ATTACK_DEFENSE" else 0.0, dtype=np.float32)
        my_turn = np.full((H, W), 1.0 if s.to_play is Player.BLACK else 0.0, dtype=np.float32)
        planes.extend([is_ad, my_turn])

        # —— grow_count 奇偶 ——
        if s.phase is Phase.ATTACK_DEFENSE:
            grow_parity = np.full((H, W), 1, dtype=np.float32)
        else:
            grow_parity = np.full((H, W), float(s.grow_count % 2), dtype=np.float32)
        planes.append(grow_parity)

        # —— 四轴向“>=4连”潜力（我方/对方）——
        me_ge4_axis_sum = self._ge4_axis_sum_plane(grid, me_id)
        opp_ge4_axis_sum = self._ge4_axis_sum_plane(grid, opp_id)
        planes.extend([me_ge4_axis_sum, opp_ge4_axis_sum])


        # —— 历史 k 步：逐方逐步（我方 recent1..4；对方 recent1..4）——
        # 根据全局步号 s.turn 推断每一手的落子方：第 i 手由 _player_of_move_index(i) 决定
        # 自末尾向前取 last_k 手，分别写入对应的我方/对方通道

        for j in range(1, self.last_k):
            if j > len(s.last_moves):
                planes.extend([np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)])
                continue
            past_grid = s.last_moves[-j]
            if past_grid is None:
                continue
            move_index = s.turn - j  # 第 move_index 手是第几回合，偶数回合p=1黑棋下，否则p=2白棋下
            p = 1 if move_index % 2 == 0 else 2
            opp_p = 2 if p == 1 else 1
            me = (past_grid == p).astype(np.float32)
            opp = (past_grid  == opp_p).astype(np.float32)
            planes.extend([me, opp])

        
        # —— 进度与奇偶 ——
        # turn_norm = np.full((H, W), float(s.turn) / max(1, s.cfg.max_turns), dtype=np.float32)
        # parity = np.full((H, W), 1.0 if (s.turn % 2) == 0 else 0.0, dtype=np.float32)
        # planes.extend([turn_norm, parity])

        out = np.stack(planes, axis=0)  # [C,H,W]
        assert out.shape[0] == 25, f"expect 25 planes, got {out.shape[0]}"
        return out

    @property
    def num_planes(self) -> int:
        # 固定返回 constant，以匹配上面的布局
        return 25
