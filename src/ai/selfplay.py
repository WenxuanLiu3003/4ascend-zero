from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch

from ..core.types import Player, Move
from ..core.engine import Engine
from ..core.state import GameState
from ..core.encoding import AlphaZeroStateEncoder
from .mcts import MCTS

# 8种对称增强
AUG_FUNCS = [
    lambda x: x,                               # id
    lambda x: np.rot90(x, 1, axes=(-2, -1)),  # rot90
    lambda x: np.rot90(x, 2, axes=(-2, -1)),
    lambda x: np.rot90(x, 3, axes=(-2, -1)),
    lambda x: np.flip(x, axis=-1),            # hflip
    lambda x: np.flip(x, axis=-2),            # vflip
    lambda x: np.flip(np.rot90(x, 1, axes=(-2, -1)), axis=-1),
    lambda x: np.flip(np.rot90(x, 1, axes=(-2, -1)), axis=-2),
]

class SelfPlay:
    def __init__(self, model, encoder: AlphaZeroStateEncoder, engine: Engine,
                 board_size: int = 9, sims: int = 400, c_puct: float = 2.0,
                 device: str = "cpu", use_tree_reuse: bool = False):
        self.model = model
        self.encoder = encoder
        self.engine = engine
        self.size = board_size
        self.sims = sims
        self.c_puct = c_puct
        self.device = device
        self.use_tree_reuse = use_tree_reuse

    def play_one(self, init_state: GameState, temp_steps: int = 30,
             dir_alpha: float = 0.3, dir_eps: float = 0.25) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """返回 (state_planes, pi, z) 列表；z 从当前执手视角 ∈ {+1,-1}。
        state_planes 使用 encoder.encode(state, as_player=state.to_play)。"""
        data = []
        s = init_state
        mcts = MCTS(self.model, self.encoder, self.engine, self.size, self.c_puct, self.sims,
                    dirichlet_alpha=dir_alpha, dirichlet_eps=dir_eps, device=self.device, reuse_tree=self.use_tree_reuse)
        step_idx = 0
        prev_root = None
        last_action = None

        while not s.is_terminal():
            if self.use_tree_reuse:
                pi, root = mcts.run(s, prev_root=prev_root, last_action=last_action, turn_related_sim=40, turn_related_sim_coef=0.5)
            else:
                pi, root = mcts.run(s, turn_related_sim=40, turn_related_sim_coef=0.5)

            planes = self.encoder.encode(s, as_player=s.to_play)  # [C,H,W]

            # —— 额外安全：用“当前真实棋盘”的合法掩码再次过滤 π —— #
            legal_mask = (s.board.grid == 0).astype(np.float32).reshape(-1)
            pi = pi * legal_mask
            ssum = pi.sum()
            if ssum <= 1e-8:
                # 极端：无合法动作
                print("[Warning] MCTS output near-zero probability sum.")
                print("psum = %.20f" % ssum)
                # break
            pi /= ssum

            # 温度控制（采样/贪心）
            if step_idx < temp_steps:
                a = np.random.choice(self.size * self.size, p=pi)
            else:
                a = int(np.argmax(pi))

            # 再次防御式检查：若因数值误差导致选择到非法点，则回退为合法 argmax
            r, c = divmod(a, self.size)
            if s.board.grid[r, c] != 0:
                legal_indices = np.where(legal_mask > 0)[0]
                if legal_indices.size == 0:
                    break
                a = int(legal_indices[np.argmax(pi[legal_indices])])
                r, c = divmod(a, self.size)
                if s.board.grid[r, c] != 0:
                    # 仍非空，放弃该局（极少数保护分支）
                    break

            # —— 计算“扣血微奖励” aux_r ——
            hp_before_black, hp_before_white = int(s.hp[0]), int(s.hp[1])
            to_play = s.to_play # 当前执手

            # 真正落子
            s = self.engine.step(s, Move(r, c))

            # HP 变化（对手被扣血的量；归一化到 [0,1]）
            hp_after_black, hp_after_white = int(s.hp[0]), int(s.hp[1])
            if to_play is Player.BLACK:
                opp_delta = max(0, hp_before_white - hp_after_white)
            else:
                opp_delta = max(0, hp_before_black - hp_after_black)
            aux_r = float(opp_delta) / max(1, s.cfg.hp_max)

            data.append((planes, pi, 0, aux_r))  # z 暂存 0，赛后再填
            step_idx += 1

            if self.use_tree_reuse:
                prev_root, last_action = root, a

        # 终局 z：胜方为 +1，负方为 -1（从各自落子时的视角）
        if (s.hp <= 0).any():
            loser_idx = 0 if s.hp[0] <= 0 else 1
        else:
            if s.hp[0] < s.hp[1]: loser_idx = 0
            elif s.hp[1] < s.hp[0]: loser_idx = 1
            else: loser_idx = -1  # 平局
        cur_player = init_state.to_play
        for i in range(len(data)):
            if loser_idx < 0:
                z = 0
            else:
                loser = Player.BLACK if loser_idx == 0 else Player.WHITE
                z = -1 if cur_player == loser else +1
            planes, pi, _, aux_r = data[i]
            data[i] = (planes, pi, z, aux_r)
            cur_player = Player.WHITE if cur_player == Player.BLACK else Player.BLACK
        return data

    @staticmethod
    def augment(sample: Tuple[np.ndarray, np.ndarray, int]) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        x, pi, z, aux = sample
        H = int(np.sqrt(len(pi)))
        pi_map = pi.reshape(H, H)
        out = []
        for f in AUG_FUNCS:
            x2 = f(x.copy())
            pi2 = f(pi_map.copy()).reshape(-1)
            out.append((x2, pi2, z, aux))
        return out