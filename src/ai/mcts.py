# ──────────────────────────────────────────────────────────────────────────────
# File: src/ai/mcts.py
# 说明：
#   - 轻量 AlphaZero 风格 MCTS，支持“根节点树复用”（可开关）。
#   - 仅在根注入 Dirichlet 噪声增强探索。
#   - 与现有引擎/编码保持解耦：通过 Engine.step & Encoder.encode 前进/评估。
#   - 返回：π（按访问计数归一化）与 root（供可选复用）。
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
# 结点定义（仅保存搜索需要的统计量）
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Node:
    prior: float                    # 先验概率 P(a|s)
    to_play: Player                 # 该结点局面下的执手（用于价值符号翻转）
    N: int = 0                      # 访问次数
    W: float = 0.0                  # 累计价值（从当前结点视角）
    Q: float = 0.0                  # 平均价值 W/N
    children: Dict[int, "Node"] = None
    is_expanded: bool = False

    def __post_init__(self):
        if self.children is None:
            self.children = {}


# ──────────────────────────────────────────────────────────────────────────────
# MCTS 主体
# ──────────────────────────────────────────────────────────────────────────────
class MCTS:
    def __init__(self,
                 model,
                 encoder: AlphaZeroStateEncoder,
                 engine: Engine,
                 board_size: int = 9,
                 c_puct: float = 2.0,
                 sims: int = 400,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_eps: float = 0.25,
                 device: str = "cpu",
                 reuse_tree: bool = False):
        """
        参数：
          - model: 策略-价值网络（forward: x->[p_logits, v]）
          - encoder: AlphaZeroStateEncoder（把 GameState 编到张量）
          - engine: 规则引擎（step/状态推进）
          - board_size: 棋盘边长（动作空间=board_size^2）
          - c_puct: PUCT 探索常数
          - sims: 每步模拟次数
          - dirichlet_alpha/eps: 根噪声参数
          - device: 推理设备
          - reuse_tree: 是否启用“根节点树复用”（上一手选中子节点→下一手的根）
                        注：你的规则含“植物随机刷新”，开启可能有轻微偏差，请按需使用。
        """
        self.model = model
        self.model.eval()  # 推理模式（关闭 dropout/bn 训练行为）
        self.encoder = encoder
        self.engine = engine
        self.size = board_size
        self.c_puct = c_puct
        self.sims = sims
        self.dir_alpha = dirichlet_alpha
        self.dir_eps = dirichlet_eps
        self.device = device
        self.reuse_tree = reuse_tree

    # ----------------------------------------------------------------------
    # 基础：动作索引 & 合法掩码
    # ----------------------------------------------------------------------
    def _legal_mask(self, s: GameState) -> np.ndarray:
        """
        返回一维合法掩码（H*W）。当前规则：空位可落（无论 NORMAL / ATTACK_DEFENSE），
        其余非法置 0。
        """
        return (s.board.grid == 0).astype(np.float32).reshape(-1)

    # ----------------------------------------------------------------------
    # 网络评估（softmax 后的策略 & 标量价值），自动应用合法掩码并归一化
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
    # 入口：执行 sims 次模拟；返回 π（按访问次数归一化）与根
    # 可选：当 reuse_tree=True 且提供 prev_root/last_action 时复用其子为新根
    # ----------------------------------------------------------------------
    def run(self, root_state: GameState,
            prev_root: Optional[Node] = None,
            last_action: Optional[int] = None,
            turn_related_sim: Optional[int] = -1, 
            turn_related_sim_coef: Optional[int] = 0.5) -> Tuple[np.ndarray, Node]:
        root_player = root_state.to_play

        # —— 根构建（可选复用）——
        if self.reuse_tree and (prev_root is not None) and (last_action is not None) and (last_action in prev_root.children):
            # 复用上一手选中动作对应的子节点作为新根
            root = prev_root.children[last_action]
        else:
            # 新建根并按网络策略扩展一次
            root = Node(prior=1.0, to_play=root_player)
            p, _ = self._policy_value(root_state, as_player=root_player)
            self._expand(root, root_state, p)

        # —— 根注入 Dirichlet 噪声（鼓励探索）——
        noise = np.random.dirichlet([self.dir_alpha] * (self.size * self.size))
        for a, child in root.children.items():
            # child.prior ← (1-ε)·prior + ε·noise
            child.prior = (1 - self.dir_eps) * child.prior + self.dir_eps * float(noise[a])

        # —— 执行多次模拟 —— 
        num_sim = self.sims
        if turn_related_sim > 0 and root_state.turn >= turn_related_sim:
            num_sim = int(self.sims * turn_related_sim_coef)
        for _ in range(num_sim):
            self._simulate(root_state, root)

        # —— 访问频次 → π —— 
        pi = np.zeros(self.size * self.size, dtype=np.float32)
        for a, child in root.children.items():
            pi[a] = child.N
        if pi.sum() > 0:
            pi = pi / pi.sum()
        return pi, root

    # ----------------------------------------------------------------------
    # 单次模拟：选择 → 前进 → 评估/扩展 → 回传
    # ----------------------------------------------------------------------
    def _simulate(self, state: GameState, node: Node):
        path = []          # 记录 (node, action) 路径用于回传
        s = state
        n = node

        # 1) Selection：沿着 argmax(Q + U) 走到叶子
        while n.is_expanded and len(n.children) > 0:
            a, n_next = self._select_child(n)
            path.append((n, a))
            r, c = divmod(a, self.size)
            s = self.engine.step(s, Move(r, c))  # 基于引擎推进到后继局面
            n = n_next
            if s.is_terminal():
                break

        # 2) Evaluation/Expansion：未终局则网络评估并创建子节点；终局则设 v
        if not s.is_terminal():
            p, v = self._policy_value(s, as_player=s.to_play)
            self._expand(n, s, p)
        else:
            # 简单的终局价值：从当前结点视角，胜=+1/负=-1/和=0
            # 若你的 GameState 提供 winner 判定，可直接替换以下逻辑。
            loser_idx = -1
            if (s.hp <= 0).any():
                # hp<=0 的一方为败者
                loser_idx = 0 if s.hp[0] <= 0 else 1
            elif s.hp[0] < s.hp[1]:
                loser_idx = 0
            elif s.hp[1] < s.hp[0]:
                loser_idx = 1
            else:
                loser_idx = -1
            
            if loser_idx >= 0:
                loser = Player.BLACK if loser_idx == 0 else Player.WHITE
                winner = Player.WHITE if loser is Player.BLACK else Player.BLACK
                v = 1.0 if winner == n.to_play else -1.0
            else:
                v = 0.0

        # 3) Backup：沿路径回传，逐层交替取反（对手视角）
        self._backup(path, v)

    # ----------------------------------------------------------------------
    # 在叶子按网络先验扩展所有合法子
    # ----------------------------------------------------------------------
    def _expand(self, node: Node, state: GameState, prior_probs: np.ndarray):
        node.is_expanded = True
        mask = self._legal_mask(state)
        legal_actions = np.where(mask > 0.0)[0]
        for a in legal_actions:
            node.children[a] = Node(prior=float(prior_probs[a]), to_play=state.to_play)

    # ----------------------------------------------------------------------
    # 选择子节点：PUCT = Q + c_puct * P * sqrt(sumN) / (1+N)
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
    # 回传：沿路径更新 N/W/Q；下一层从对手视角认为价值取负
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
