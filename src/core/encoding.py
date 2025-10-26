from __future__ import annotations
import numpy as np
from typing import List

from .state import GameState
from .types import Player

class AlphaZeroStateEncoder:
    """
    将 GameState 编码为 [C,H,W] 的多通道张量，用于策略-价值网络输入。

    通道定义（总计 20 个；以 as_player 视角构建）：
    0: 我方棋（one-hot）
    1: 对方棋（one-hot）
    2: 攻方“可被无效化”的连子掩码（ascend才有）
    3: 植物“数量”通道（float，计数，0/1/2 ...）
    4: 我方 HP 归一化平铺（/hp_max）
    5: 对方 HP 归一化平铺（/hp_max）
    6: 是否处于ascend阶段（ascend）
    7: 当前是否轮到黑棋（my_turn）
    8 - 21: 历史7步的[当前棋子分布, 对手棋子分布]
    """

    def __init__(self, last_k: int = 8):
        # 虽然 last_k 可调，但当前通道布局固定为 4；若要更长历史，可同步修改下方拼接逻辑
        self.last_k = last_k

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
        assert out.shape[0] == 22, f"expect 22 planes, got {out.shape[0]}"
        return out

    @property
    def num_planes(self) -> int:
        # 固定返回 constant，以匹配上面的布局
        return 22