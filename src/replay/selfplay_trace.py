from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import copy
import numpy as np
import torch

from ..core.rules import RulesConfig
from ..core.board import Board
from ..core.state import GameState
from ..core.types import Player, Move, Phase
from ..core.engine import Engine
from ..core.encoding import AlphaZeroStateEncoder
from ..ai.model import PolicyValueNet
from ..ai.mcts import MCTS

@dataclass
class TraceStep:
    """一帧回放所需的最小信息（避免直接存复杂对象，便于序列化/回放）。"""
    turn: int                          # 全局步号
    to_play: int                       # 当前行动方（1=BLACK, 2=WHITE）
    last_move: Optional[Tuple[int,int]]# 上一步落点（用于高亮）
    grid: np.ndarray                   # 棋子网格 (H,W)，值∈{0,1,2}
    plants: np.ndarray                 # 植物计数 (H,W)，uint16
    hp: Tuple[int,int]                 # (黑HP, 白HP)
    phase: str                         # 'NORMAL' / 'ATTACK_DEFENSE'
    attack_chain_mask: Optional[np.ndarray]  # 攻方连子掩码
    plant_events: Dict[str, Any]       # {'spawn':[(r,c,delta),...], 'clear':[(r,c,delta-)]}


def _detect_plant_events(prev: Optional[np.ndarray], cur: np.ndarray) -> Dict[str, Any]:
    """比较前后两帧的植物矩阵，记录刷新/减少事件。
    - spawn: delta>0 的位置（含 +1 或 +2）
    - clear: delta<0 的位置（攻防后清除、或被刷新覆盖的边界情况）
    """
    events = {"spawn": [], "clear": []}
    if prev is None:
        # 首帧没有事件
        return events
    diff = cur.astype(np.int32) - prev.astype(np.int32)
    rs, cs = np.where(diff > 0)
    for r, c in zip(rs, cs):
        events["spawn"].append((int(r), int(c), int(diff[r, c])))
    rs, cs = np.where(diff < 0)
    for r, c in zip(rs, cs):
        events["clear"].append((int(r), int(c), int(diff[r, c])))
    return events


def generate_selfplay_trace(cfg: RulesConfig, model: PolicyValueNet, device: str = "cpu",
                            sims: int = 400, temp_steps: int = 20,
                            dir_alpha: float = 0.3, dir_eps: float = 0.25,
                            use_tree_reuse: bool = False) -> List[TraceStep]:
    """用当前模型打一整局自对弈并生成轨迹（TraceStep 列表）。
    - 不管训练时是否启用树复用，这里可用开关 `use_tree_reuse` 控制；考虑到植物随机刷新，默认 False。
    - 每一步都会保存：棋子、植物、HP、阶段、攻方掩码、落点，以及基于植物矩阵前后差分的事件（刷新/清除）。
    """
    engine = Engine(win_k=cfg.win_k)
    encoder = AlphaZeroStateEncoder(last_k=4)
    size = cfg.board_size

    # 进入推理模式以提速
    model.eval()

    # 初始状态
    state = GameState(cfg=cfg, board=Board(size))
    trace: List[TraceStep] = []
    prev_plants = None

    # MCTS 对象（每手都会复用同一个实例，但可选择是否复用根）
    mcts = MCTS(model, encoder, engine, board_size=size, c_puct=2.0, sims=sims,
                dirichlet_alpha=dir_alpha, dirichlet_eps=dir_eps, device=device)
    prev_root = None
    last_action = None

    step_idx = 0
    while not state.is_terminal():
        # 搜索：根据是否复用树选择调用方式
        pi, _ = mcts.run(state)

        # 再次用实时合法掩码过滤 π，确保安全
        legal_mask = (state.board.grid == 0).astype(np.float32).reshape(-1)
        pi = pi * legal_mask
        ssum = pi.sum()
        if ssum <= 1e-8:
            break
        pi /= ssum

        # 温度：前 temp_steps 采样，之后 argmax
        if step_idx < temp_steps:
            a = int(np.random.choice(size * size, p=pi))
        else:
            a = int(np.argmax(pi))
        r, c = divmod(a, size)
        # 防御式回退
        if state.board.grid[r, c] != 0:
            legal_indices = np.where(legal_mask > 0)[0]
            if legal_indices.size == 0:
                break
            a = int(legal_indices[np.argmax(pi[legal_indices])])
            r, c = divmod(a, size)
            if state.board.grid[r, c] != 0:
                break

        # 记录“当前帧”之前的静态信息（用于在这一步走子前展示现状）
        step = TraceStep(
            turn=state.turn,
            to_play=1 if state.to_play is Player.BLACK else 2,
            last_move=(state.last_move.r, state.last_move.c) if state.last_move else None,
            grid=state.board.grid.copy(),
            plants=state.board.plants.copy(),
            hp=(int(state.hp[0]), int(state.hp[1])),
            phase=state.phase.name,
            attack_chain_mask=(state.attack_chain_mask.copy() if state.attack_chain_mask is not None else None),
            plant_events=_detect_plant_events(prev_plants, state.board.plants),
        )
        trace.append(step)
        prev_plants = state.board.plants.copy()

        # 真正落子前进
        state = engine.step(state, Move(r, c))
        step_idx += 1

        # 树复用需要的上下文
        prev_root, last_action = (root, a) if use_tree_reuse else (None, None)

    # 最后一帧（终局态）
    step = TraceStep(
        turn=state.turn,
        to_play=1 if state.to_play is Player.BLACK else 2,
        last_move=(state.last_move.r, state.last_move.c) if state.last_move else None,
        grid=state.board.grid.copy(),
        plants=state.board.plants.copy(),
        hp=(int(state.hp[0]), int(state.hp[1])),
        phase=state.phase.name,
        attack_chain_mask=(state.attack_chain_mask.copy() if state.attack_chain_mask is not None else None),
        plant_events=_detect_plant_events(prev_plants, state.board.plants),
    )
    trace.append(step)

    return trace
