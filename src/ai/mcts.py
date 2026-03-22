# ──────────────────────────────────────────────────────────────────────────────
# File: src/ai/mcts.py
# The MCTS implementation for 4ascend-zero
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch

from ..core.types import Player, Move
from ..core.engine import Engine
from ..core.state import GameState
from ..core.encoding import AlphaZeroStateEncoder


# ──────────────────────────────────────────────────────────────────────────────
# Node definition
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Node:
    prior: float                    # P(a|s)
    to_play: Player                 # the player who takes action at this node
    N: int = 0                      # visit counts
    W: float = 0.0                  # cumulative value
    Q: float = 0.0                  # average value W/N
    children: Dict[int, "Node"] = None
    is_expanded: bool = False

    def __post_init__(self):
        if self.children is None:
            self.children = {}


def _other(p: Player) -> Player:
    return Player.WHITE if p is Player.BLACK else Player.BLACK


# ──────────────────────────────────────────────────────────────────────────────
# MCTS
# ──────────────────────────────────────────────────────────────────────────────
class MCTS:
    def __init__(self,
                 model,
                 encoder: AlphaZeroStateEncoder,
                 engine: Engine,
                 board_size: int = 9,
                 c_puct: float = 2.0,
                 sims: int = 400,
                 dirichlet_alpha: float = 0.35,
                 dirichlet_eps: float = 0.25,
                 device: str = "cpu",
                 reuse_tree: bool = False):
        """
        parameters:
          - model: the policy-value network for evaluation
          - encoder: AlphaZeroStateEncoder that encodes GameState to inputs for the model
          - engine: rule engine to step the GameState
          - board_size: 
          - c_puct: hyperparameter controlling the exploration strength in UCT formula
          - sims: the number of simulations to run for each move
          - dirichlet_alpha/eps: hyperparameters controlling the Dirichlet noise to the root node to encourage exploration
          - device: CPU or GPU for model inference
          - reuse_tree: whether to reuse the search tree across moves (default False since the plant refreshing is random)
        """
        self.model = model
        self.model.eval()
        self.encoder = encoder
        self.engine = engine
        self.size = board_size
        self.c_puct = c_puct
        self.sims = sims
        self.dir_alpha = dirichlet_alpha
        self.dir_eps = dirichlet_eps
        self.device = device
        self.reuse_tree = reuse_tree

    def _legal_mask(self, s: GameState) -> np.ndarray:
        """
        返回一维合法掩码（H*W）。当前规则：空位可落（无论 NORMAL / ATTACK_DEFENSE），
        其余非法置 0。
        """
        return (s.board.grid == 0).astype(np.float32).reshape(-1)

    # ----------------------------------------------------------------------
    # use model to evaluate the policy and value for a given state from as_player's perspective
    # ----------------------------------------------------------------------
    def _policy_value(self, s: GameState, as_player: Player) -> Tuple[np.ndarray, float]:
        with torch.inference_mode():  # 比 no_grad 更彻底禁用 autograd
            x = torch.from_numpy(self.encoder.encode(s, as_player=as_player)).unsqueeze(0).to(self.device)
            p_logits, v = self.model(x)               # p_logits: [1, H*W], v: [1]
            p = torch.softmax(p_logits, dim=-1).cpu().numpy()[0]
            v = float(v.item())
        mask = self._legal_mask(s)
        p = p * mask
        ssum = p.sum()
        if ssum <= 1e-8:
            # 如果因数值或禁手导致全 0，则在合法点上均匀分布
            n = mask.sum()
            if n > 0:
                p = mask / n
        else:
            p /= ssum
        return p, v

    # ----------------------------------------------------------------------
    # Run the MCTS Algorithm and return the final policy π and the root node
    # ----------------------------------------------------------------------
    def run(self, root_state: GameState,
            prev_root: Optional[Node] = None,
            last_action: Optional[int] = None,
            turn_related_sim: Optional[int] = -1, 
            turn_related_sim_coef: Optional[int] = 0.5) -> Tuple[np.ndarray, Node]:
        root_player = root_state.to_play

        if self.reuse_tree and (prev_root is not None) and (last_action is not None) and (last_action in prev_root.children):
            root = prev_root.children[last_action]
        else:
            # by default, we do not use tree reusing, and we set a new root for each move
            root = Node(prior=1.0, to_play=root_player)
            p, v = self._policy_value(root_state, as_player=root_player)
            self._expand(root, root_state, p)

        # add Dirichlet noise to the root node's priors to encourage exploration (only for the first move in self-play)
        noise = np.random.dirichlet([self.dir_alpha] * (self.size * self.size))
        for a, child in root.children.items():
            child.prior = (1 - self.dir_eps) * child.prior + self.dir_eps * float(noise[a])

        # Simulations
        num_sim = self.sims
        if turn_related_sim > 0 and root_state.turn >= turn_related_sim:
            num_sim = int(self.sims * turn_related_sim_coef)
        for _ in range(num_sim):
            self._simulate(root_state, root)

        # return the policy as the frequency in proportional to visit counts N.
        pi = np.zeros(self.size * self.size, dtype=np.float32)
        for a, child in root.children.items():
            pi[a] = child.N
        if pi.sum() > 0:
            pi = pi / pi.sum()
        return pi, root

    # ----------------------------------------------------------------------
    # Single simulation
    # ----------------------------------------------------------------------
    def _simulate(self, state: GameState, node: Node):
        path = []          # the exploration path along the tree
        s = state
        n = node

        # 1) Selection：select actions according to UCT until reaching a leaf node (unexpanded or terminal)
        while n.is_expanded and len(n.children) > 0:
            a, n_next = self._select_child(n)
            path.append((n, a))
            r, c = divmod(a, self.size)
            s = self.engine.step(s, Move(r, c)) 
            n = n_next
            if s.is_terminal():
                break

        # 2) Evaluation/Expansion：if not terminal, evaluate the leaf node with the model and expand it; if terminal, directly use the game result as the value
        if not s.is_terminal():
            p, v = self._policy_value(s, as_player=s.to_play)
            self._expand(n, s, p)
        else:
            # If terminal, determine the winner and assign value v from the perspective of the current player
            loser_idx = -1
            if (s.hp <= 0).any():
                loser_idx = 0 if s.hp[0] <= 0 else 1
            elif s.hp[0] < s.hp[1]:
                loser_idx = 0
            elif s.hp[1] < s.hp[0]:
                loser_idx = 1
            else:
                black_cnt = int(np.sum(s.board.grid == 1))
                white_cnt = int(np.sum(s.board.grid == 2))
                if black_cnt > white_cnt:
                    loser_idx = 1
                elif white_cnt > black_cnt:
                    loser_idx = 0
                else:
                    loser_idx = -1
            
            if loser_idx >= 0:
                loser = Player.BLACK if loser_idx == 0 else Player.WHITE
                winner = Player.WHITE if loser is Player.BLACK else Player.BLACK
                v = 1.0 if winner == n.to_play else -1.0
            else:
                v = 0.0

        # 3) Backup the value along the path, flipping the sign for the opponent's perspective
        self._backup(path, -v)

    # ----------------------------------------------------------------------
    # when reaching the leaf node, expand it by adding child nodes for all legal actions, and set their priors according to the model's policy output
    # ----------------------------------------------------------------------
    def _expand(self, node: Node, state: GameState, prior_probs: np.ndarray):
        node.is_expanded = True
        mask = self._legal_mask(state)
        legal_actions = np.where(mask > 0.0)[0]
        for a in legal_actions:
            node.children[a] = Node(prior=float(prior_probs[a]), to_play=_other(state.to_play))

    # ----------------------------------------------------------------------
    # Criterion to select action: Q + c_puct * P * sqrt(sumN) / (1+N)
    # ----------------------------------------------------------------------
    def _select_child(self, node: Node) -> Tuple[int, Node]:
        sumN = max(1, sum(child.N for child in node.children.values()))
        best = (-10**9, None, None)  # (score, action, child)
        for a, child in node.children.items():
            u = self.c_puct * child.prior * (np.sqrt(sumN) / (1 + child.N))
            score = child.Q + u
            if score > best[0]:
                best = (score, a, child)
        return best[1], best[2]

    # ----------------------------------------------------------------------
    # backup the value along the path, flipping the sign for the opponent's perspective
    # ----------------------------------------------------------------------
    def _backup(self, path, value: float):
        v = value
        # for node, _ in reversed(path):
        #     node.N += 1
        #     node.W += v
        #     node.Q = node.W / node.N
        #     v = -v  # 轮到对手视角，符号翻转
        for parent, a in reversed(path):
            child = parent.children[a]
            child.N += 1
            child.W += v
            child.Q = child.W / child.N
            v = -v
