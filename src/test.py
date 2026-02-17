from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .ai.model import PolicyValueNet
from .ai.mcts import MCTS
from .core.board import Board
from .core.engine import Engine
from .core.rules import RulesConfig
from .core.state import GameState
from .core.types import Player, Move
from .core.encoding import AlphaZeroStateEncoder
from .utils.checkpoint import load_checkpoint

__IF__HPC__ = "SLURM_JOB_ID" in os.environ


def _list_model_files_by_mtime(save_path: str) -> List[str]:
    files = []
    for name in os.listdir(save_path):
        if name.endswith(".pt"):
            path = os.path.join(save_path, name)
            if os.path.isfile(path):
                files.append(path)
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def _determine_winner(state: GameState) -> Optional[Player]:
    if (state.hp <= 0).any():
        loser_idx = 0 if state.hp[0] <= 0 else 1
    else:
        if state.hp[0] < state.hp[1]:
            loser_idx = 0
        elif state.hp[1] < state.hp[0]:
            loser_idx = 1
        else:
            loser_idx = -1
    if loser_idx < 0:
        return None
    loser = Player.BLACK if loser_idx == 0 else Player.WHITE
    return Player.WHITE if loser is Player.BLACK else Player.BLACK


def _select_action(pi: np.ndarray, state: GameState) -> int:
    action = int(np.argmax(pi))
    r, c = divmod(action, state.cfg.board_size)
    if state.board.grid[r, c] != 0:
        legal_mask = (state.board.grid == 0).reshape(-1)
        legal_indices = np.where(legal_mask > 0)[0]
        if legal_indices.size == 0:
            return action
        action = int(legal_indices[np.argmax(pi[legal_indices])])
    return action


def play_one_game(
    model_black: PolicyValueNet,
    model_white: PolicyValueNet,
    cfg: RulesConfig,
    sims: int,
    device: str,
    c_puct: float = 2.0,
) -> Optional[Player]:
    encoder = AlphaZeroStateEncoder(last_k=8)
    engine = Engine(win_k=cfg.win_k)
    mcts_black = MCTS(model_black, encoder, engine, board_size=cfg.board_size,
                      c_puct=c_puct, sims=sims, dirichlet_eps=0.0, device=device,
                      reuse_tree=False)
    mcts_white = MCTS(model_white, encoder, engine, board_size=cfg.board_size,
                      c_puct=c_puct, sims=sims, dirichlet_eps=0.0, device=device,
                      reuse_tree=False)

    state = GameState(cfg=cfg, board=Board(cfg.board_size), to_play=Player.BLACK)
    prev_root = {Player.BLACK: None, Player.WHITE: None}
    last_action = {Player.BLACK: None, Player.WHITE: None}

    while not state.is_terminal():
        if state.to_play is Player.BLACK:
            mcts = mcts_black
            prev = prev_root[Player.BLACK]
            last = last_action[Player.BLACK]
        else:
            mcts = mcts_white
            prev = prev_root[Player.WHITE]
            last = last_action[Player.WHITE]

        pi, root = mcts.run(state, prev_root=prev, last_action=last,
                            turn_related_sim=-1, turn_related_sim_coef=0.5)
        action = _select_action(pi, state)
        r, c = divmod(action, cfg.board_size)
        state = engine.step(state, Move(r, c))

        if state.to_play is Player.WHITE:
            prev_root[Player.BLACK] = root
            last_action[Player.BLACK] = action
        else:
            prev_root[Player.WHITE] = root
            last_action[Player.WHITE] = action

    return _determine_winner(state)


def evaluate_models(
    model_paths: List[str],
    cfg: RulesConfig,
    num_game: int,
    sims: int,
    device: str,
) -> Tuple[Dict[str, float], str]:
    encoder = AlphaZeroStateEncoder(last_k=8)
    models: Dict[str, PolicyValueNet] = {}
    for path in model_paths:
        model = PolicyValueNet(in_planes=encoder.num_planes, board_size=cfg.board_size).to(device)
        load_checkpoint(path, model, optimizer=None, map_location=device)
        model.eval()
        models[path] = model

    scores = {path: 0.0 for path in model_paths}
    if len(model_paths) != 2:
        raise ValueError("evaluate_models expects exactly two model paths.")

    path_a, path_b = model_paths
    model_a = models[path_a]
    model_b = models[path_b]
    for g in tqdm(range(num_game), desc="Eval games", unit="game"):
        start_t = time.perf_counter()
        if g % 2 == 0:
            winner = play_one_game(model_a, model_b, cfg, sims, device)
            if winner is Player.BLACK:
                scores[path_a] += 1.0
            elif winner is Player.WHITE:
                scores[path_b] += 1.0
            else:
                scores[path_a] += 0.5
                scores[path_b] += 0.5
        else:
            winner = play_one_game(model_b, model_a, cfg, sims, device)
            if winner is Player.BLACK:
                scores[path_b] += 1.0
            elif winner is Player.WHITE:
                scores[path_a] += 1.0
            else:
                scores[path_a] += 0.5
                scores[path_b] += 0.5
        elapsed = time.perf_counter() - start_t
        tqdm.write(f"[test] game {g+1}/{num_game} finished in {elapsed:.2f}s")

    best_path = max(scores.items(), key=lambda kv: kv[1])[0]
    return scores, best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints via 1v1 matches.")
    # default_path = "/insomnia001/depts/free/users/wl3003/4ascend-model/checkpoints"
    default_path = "checkpoints"
    parser.add_argument("--savePath", type=str, default=default_path, help="path to checkpoints")
    parser.add_argument("--num_game", type=int, default=9, help="games per pair (1v1)")
    parser.add_argument("--sim", type=int, default=1200, help="MCTS simulations per move")
    parser.add_argument("--board_size", type=int, default=9)
    parser.add_argument("--win_k", type=int, default=4)
    parser.add_argument("--hp_max", type=int, default=6)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RulesConfig(board_size=args.board_size, win_k=args.win_k, hp_max=args.hp_max)

    if not os.path.isdir(args.savePath):
        raise FileNotFoundError(f"savePath not found: {args.savePath}")
    model_paths = _list_model_files_by_mtime(args.savePath)
    if len(model_paths) == 0:
        raise RuntimeError("No model files found to evaluate.")
    if len(model_paths) == 1:
        print(os.path.basename(model_paths[0]))
        return
    # Compare only the most recent and second most recent checkpoints.
    model_paths = model_paths[:2]

    print(f"[test] comparing: {os.path.basename(model_paths[0])} vs {os.path.basename(model_paths[1])}")
    scores, best_path = evaluate_models(
        model_paths=model_paths,
        cfg=cfg,
        num_game=int(args.num_game),
        sims=int(args.sim),
        device=device,
    )

    for path, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{os.path.basename(path)}\t{score:.2f}")
    print(os.path.basename(best_path))


if __name__ == "__main__":
    main()
