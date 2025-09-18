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
      2: 空位（one-hot）
      3: 植物“数量”通道（float，计数，0/1/2 ...）
      4: 植物“存在”二值通道（plants>0）
      5: 是否处于攻防阶段（ATTACK_DEFENSE）
      6: 当前是否轮到我（my_turn）
      7: 我方最近第1步落点
      8: 我方最近第2步落点
      9: 我方最近第3步落点
      10: 我方最近第4步落点
      11: 对方最近第1步落点
      12: 对方最近第2步落点
      13: 对方最近第3步落点
      14: 对方最近第4步落点
      15: 我方 HP 归一化平铺（/hp_max）
      16: 对方 HP 归一化平铺（/hp_max）
      17: 攻方“可被无效化”的连子掩码（攻防期才有）
      18: 总步数归一化平铺（turn/max_turns）
      19: 回合奇偶平铺（(turn % 2)==0 -> 1.0 else 0.0）

    注：历史落点采用“逐方逐步”解耦（k=4），有利于网络学习攻守节奏；
        如需更长历史，可提高 `last_k` 并相应扩展通道数（本类固定输出 20 个通道）。
    """

    def __init__(self, last_k: int = 4):
        # 虽然 last_k 可调，但当前通道布局固定为 4；若要更长历史，可同步修改下方拼接逻辑
        self.last_k = last_k

    def _player_of_move_index(self, move_index: int) -> Player:
        """给定全局第 move_index 手（从 1 开始），返回该手棋的执手方。
        假设整局从 BLACK 先手并交替行棋（本引擎满足）。"""
        return Player.BLACK if ((move_index - 1) % 2 == 0) else Player.WHITE

    def encode(self, s: GameState, as_player: Player) -> np.ndarray:
        H, W = s.board.grid.shape
        planes: List[np.ndarray] = []

        grid = s.board.grid
        me_id = 1 if as_player is Player.BLACK else 2
        opp_id = 2 if as_player is Player.BLACK else 1

        # —— 基础棋面 ——
        me = (grid == me_id).astype(np.float32)
        opp = (grid == opp_id).astype(np.float32)
        empty = (grid == 0).astype(np.float32)
        planes.extend([me, opp, empty])

        # —— 植物：数量 + 存在二值 ——
        plants_count = s.board.plants.astype(np.float32)
        plants_mask = (s.board.plants > 0).astype(np.float32)
        planes.extend([plants_count, plants_mask])

        # —— 阶段与行动方 ——
        is_ad = np.full((H, W), 1.0 if s.phase.name == "ATTACK_DEFENSE" else 0.0, dtype=np.float32)
        my_turn = np.full((H, W), 1.0 if s.to_play is as_player else 0.0, dtype=np.float32)
        planes.extend([is_ad, my_turn])

        # —— 历史 k 步：逐方逐步（我方 recent1..4；对方 recent1..4）——
        # 根据全局步号 s.turn 推断每一手的落子方：第 i 手由 _player_of_move_index(i) 决定
        # 自末尾向前取 last_k 手，分别写入对应的我方/对方通道
        me_hist = [np.zeros((H, W), dtype=np.float32) for _ in range(self.last_k)]
        opp_hist = [np.zeros((H, W), dtype=np.float32) for _ in range(self.last_k)]
        for j in range(1, self.last_k + 1):
            if j > len(s.last_moves):
                break
            mv = s.last_moves[-j]
            if mv is None:
                continue
            move_index = s.turn - (j - 1)  # 第 move_index 手就是这一步
            p = self._player_of_move_index(move_index)
            is_me = (p is as_player)
            if is_me:
                me_hist[j - 1][mv.r, mv.c] = 1.0
            else:
                opp_hist[j - 1][mv.r, mv.c] = 1.0
        planes.extend(me_hist)
        planes.extend(opp_hist)

        # —— HP 归一化 ——
        hp_max = max(1, s.cfg.hp_max)
        me_idx = 0 if as_player.name == "BLACK" else 1
        opp_idx = 1 - me_idx
        me_hp = np.full((H, W), float(s.hp[me_idx]) / hp_max, dtype=np.float32)
        opp_hp = np.full((H, W), float(s.hp[opp_idx]) / hp_max, dtype=np.float32)
        planes.extend([me_hp, opp_hp])

        # —— 攻方“可被无效化”掩码 ——
        attack_mask = (
            s.attack_chain_mask.astype(np.float32)
            if s.attack_chain_mask is not None else np.zeros((H, W), dtype=np.float32)
        )
        planes.append(attack_mask)

        # —— 进度与奇偶 ——
        turn_norm = np.full((H, W), float(s.turn) / max(1, s.cfg.max_turns), dtype=np.float32)
        parity = np.full((H, W), 1.0 if (s.turn % 2) == 0 else 0.0, dtype=np.float32)
        planes.extend([turn_norm, parity])

        out = np.stack(planes, axis=0)  # [C,H,W]
        assert out.shape[0] == 20, f"expect 20 planes, got {out.shape[0]}"
        return out

    @property
    def num_planes(self) -> int:
        # 固定返回 20，以匹配上面的布局
        return 20