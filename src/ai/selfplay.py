from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch

from ..core.types import Player, Move
from ..core.engine import Engine
from ..core.state import GameState
from ..core.encoding import AlphaZeroStateEncoder
from .mcts import MCTS

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
             dir_alpha: float = 0.3, dir_eps: float = 0.25) -> List[Tuple[np.ndarray, np.ndarray, int, float, float]]:
        """
        play one game and collect training data. Returns a list of (planes, pi, z, aux_r, is_endgame) for each step in the game, where:
        - planes: the encoded game state as input to the model (C, H, W)
        - pi: the MCTS visit count distribution over actions (H*W,)
        - z: the game result from the perspective of the current player at that step (+1 win, -1 lose, 0 draw)
        - is_endgame: whether the game is in the endgame phase (after 64 moves) for that step
        """
        data = []
        s = init_state
        mcts = MCTS(self.model, self.encoder, self.engine, self.size, self.c_puct, self.sims,
                    dirichlet_alpha=dir_alpha, dirichlet_eps=dir_eps, device=self.device, reuse_tree=self.use_tree_reuse)
        step_idx = 0
        prev_root = None
        last_action = None
        end_game_explore_coef = 2
        base_c_puct = mcts.c_puct

        while not s.is_terminal():
            if np.count_nonzero(s.board.grid) > 60:
                mcts.c_puct = base_c_puct * end_game_explore_coef
            else:
                mcts.c_puct = base_c_puct

            if self.use_tree_reuse:
                pi, root = mcts.run(s, prev_root=prev_root, last_action=last_action, turn_related_sim=-1, turn_related_sim_coef=0.5)
            else:
                pi, root = mcts.run(s, turn_related_sim=-1, turn_related_sim_coef=0.5)

            # print("----------------pi for %s--------------------" % (s.to_play.name,))
            # print(pi, end=' ')
            # print("max_prob=%.4f" % max(pi))

            planes = self.encoder.encode(s, as_player=s.to_play)  # [C,H,W]

            legal_mask = (s.board.grid == 0).astype(np.float32).reshape(-1)
            pi = pi * legal_mask
            ssum = pi.sum()
            if ssum <= 1e-8:
                print("[Warning] MCTS output near-zero probability sum.")
                print("psum = %.20f" % ssum)
                # break
            pi /= ssum

            if step_idx < temp_steps:
                a = np.random.choice(self.size * self.size, p=pi)
            else:
                a = int(np.argmax(pi))

            r, c = divmod(a, self.size)
            if s.board.grid[r, c] != 0:
                legal_indices = np.where(legal_mask > 0)[0]
                if legal_indices.size == 0:
                    break
                a = int(legal_indices[np.argmax(pi[legal_indices])])
                r, c = divmod(a, self.size)
                if s.board.grid[r, c] != 0:
                    break

            # computing aux_r
            hp_before_black, hp_before_white = int(s.hp[0]), int(s.hp[1])
            to_play = s.to_play # 当前执手

            # place the stone
            s = self.engine.step(s, Move(r, c))

            # HP change
            hp_after_black, hp_after_white = int(s.hp[0]), int(s.hp[1])
            if to_play is Player.BLACK:
                opp_delta = max(0, hp_before_white - hp_after_white)
            else:
                opp_delta = max(0, hp_before_black - hp_after_black)
            aux_r = float(opp_delta) / max(1, s.cfg.hp_max)
            is_endgame = float(np.count_nonzero(s.board.grid) >= 64)

            data.append((planes, pi, 0, aux_r, is_endgame))  # z is temporarily set to 0, will be filled in after the game ends
            step_idx += 1

            if self.use_tree_reuse:
                prev_root, last_action = root, a

        # assign value when the game ends
        if (s.hp <= 0).any():
            loser_idx = 0 if s.hp[0] <= 0 else 1
        else:
            if s.hp[0] < s.hp[1]: loser_idx = 0
            elif s.hp[1] < s.hp[0]: loser_idx = 1
            else:
                black_cnt = int(np.sum(s.board.grid == 1))
                white_cnt = int(np.sum(s.board.grid == 2))
                if black_cnt > white_cnt:
                    loser_idx = 1
                elif white_cnt > black_cnt:
                    loser_idx = 0
                else:
                    loser_idx = -1
        cur_player = init_state.to_play
        for i in range(len(data)):
            if loser_idx < 0:
                z = 0
            else:
                loser = Player.BLACK if loser_idx == 0 else Player.WHITE
                z = -1 if cur_player == loser else +1
            planes, pi, _, aux_r, is_endgame = data[i]
            data[i] = (planes, pi, z, aux_r, is_endgame)
            cur_player = Player.WHITE if cur_player == Player.BLACK else Player.BLACK
        return data

    @staticmethod
    def augment(sample: Tuple[np.ndarray, np.ndarray, int, float, float]) -> List[Tuple[np.ndarray, np.ndarray, int, float, float]]:
        x, pi, z, aux, is_endgame = sample
        H = int(np.sqrt(len(pi)))
        pi_map = pi.reshape(H, H)
        out = []
        for f in AUG_FUNCS:
            x2 = f(x.copy())
            pi2 = f(pi_map.copy()).reshape(-1)
            out.append((x2, pi2, z, aux, is_endgame))
        return out
