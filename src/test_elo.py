from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .ai.mcts import MCTS
from .ai.model import PolicyValueNet
from .core.board import Board
from .core.engine import Engine
from .core.encoding import AlphaZeroStateEncoder
from .core.rules import RulesConfig
from .core.state import GameState
from .core.types import Move, Player
from .utils.checkpoint import load_checkpoint


@dataclass(frozen=True)
class MatchResult:
    path_a: str
    path_b: str
    winner_name: str
    a_wins: int
    b_wins: int
    draws: int


def _list_model_files_by_mtime(save_path: str) -> List[str]:
    files: List[str] = []
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
    mcts_black = MCTS(
        model_black,
        encoder,
        engine,
        board_size=cfg.board_size,
        c_puct=c_puct,
        sims=sims,
        dirichlet_eps=0.0,
        device=device,
        reuse_tree=False,
    )
    mcts_white = MCTS(
        model_white,
        encoder,
        engine,
        board_size=cfg.board_size,
        c_puct=c_puct,
        sims=sims,
        dirichlet_eps=0.0,
        device=device,
        reuse_tree=False,
    )

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

        pi, root = mcts.run(
            state,
            prev_root=prev,
            last_action=last,
            turn_related_sim=-1,
            turn_related_sim_coef=0.5,
        )
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


def _load_model(path: str, cfg: RulesConfig, device: str) -> PolicyValueNet:
    encoder = AlphaZeroStateEncoder(last_k=8)
    model = PolicyValueNet(in_planes=encoder.num_planes, board_size=cfg.board_size).to(device)
    load_checkpoint(path, model, optimizer=None, map_location=device)
    model.eval()
    return model


def _release_model(model: Optional[PolicyValueNet], device: str) -> None:
    if model is None:
        return
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def play_match(
    model_a: PolicyValueNet,
    model_b: PolicyValueNet,
    path_a: str,
    path_b: str,
    cfg: RulesConfig,
    sims: int,
    device: str,
    games_per_color: int,
) -> Tuple[str, int, int, int]:
    a_wins = 0
    b_wins = 0
    draws = 0

    for _ in range(games_per_color):
        winner = play_one_game(model_a, model_b, cfg, sims, device)
        if winner is Player.BLACK:
            a_wins += 1
        elif winner is Player.WHITE:
            b_wins += 1
        else:
            draws += 1

    for _ in range(games_per_color):
        winner = play_one_game(model_b, model_a, cfg, sims, device)
        if winner is Player.BLACK:
            b_wins += 1
        elif winner is Player.WHITE:
            a_wins += 1
        else:
            draws += 1

    if a_wins > b_wins:
        return os.path.basename(path_a), a_wins, b_wins, draws
    if a_wins < b_wins:
        return os.path.basename(path_b), a_wins, b_wins, draws
    return "draw", a_wins, b_wins, draws


def evaluate_all_models(
    model_paths: Sequence[str],
    cfg: RulesConfig,
    games_per_color: int,
    sims: int,
    device: str,
    output_path: Optional[str] = None,
) -> List[MatchResult]:
    if len(model_paths) < 2:
        return []

    results: List[MatchResult] = []
    total_pairs = len(model_paths) * (len(model_paths) - 1) // 2
    pair_bar = tqdm(total=total_pairs, desc="Pair matches", unit="pair")

    for idx, path_a in enumerate(model_paths[:-1]):
        model_a: Optional[PolicyValueNet] = None
        try:
            model_a = _load_model(path_a, cfg, device)
            for path_b in model_paths[idx + 1 :]:
                model_b: Optional[PolicyValueNet] = None
                start_t = time.perf_counter()
                try:
                    model_b = _load_model(path_b, cfg, device)
                    winner_name, a_wins, b_wins, draws = play_match(
                        model_a=model_a,
                        model_b=model_b,
                        path_a=path_a,
                        path_b=path_b,
                        cfg=cfg,
                        sims=sims,
                        device=device,
                        games_per_color=games_per_color,
                    )
                    result = MatchResult(
                        path_a=path_a,
                        path_b=path_b,
                        winner_name=winner_name,
                        a_wins=a_wins,
                        b_wins=b_wins,
                        draws=draws,
                    )
                    results.append(result)
                    if output_path is not None:
                        append_match_results(output_path, [result])
                    elapsed = time.perf_counter() - start_t
                    tqdm.write(
                        "[test-elo] "
                        f"{os.path.basename(path_a)} vs {os.path.basename(path_b)} "
                        f"=> {a_wins}-{b_wins} (draw {draws}) in {elapsed:.2f}s"
                    )
                finally:
                    _release_model(model_b, device)
                    pair_bar.update(1)
        finally:
            _release_model(model_a, device)

    pair_bar.close()
    return results


def append_match_results(output_path: str, match_results: Sequence[MatchResult]) -> None:
    with open(output_path, "a", encoding="utf-8") as output_file:
        for result in match_results:
            output_file.write(
                f"{os.path.basename(result.path_a)}, "
                f"{os.path.basename(result.path_b)}, "
                f"{result.winner_name}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints and record pairwise winners.")
    default_path = os.environ.get("ASCEND_CHECKPOINT_DIR", "checkpoints")
    parser.add_argument("--savePath", type=str, default=default_path, help="path to checkpoints")
    parser.add_argument(
        "--games_per_color",
        type=int,
        default=2,
        help="number of games with each model taking black once per pairing",
    )
    parser.add_argument("--sim", type=int, default=1600, help="MCTS simulations per move")
    parser.add_argument("--board_size", type=int, default=9)
    parser.add_argument("--win_k", type=int, default=4)
    parser.add_argument("--hp_max", type=int, default=6)
    parser.add_argument(
        "--output",
        type=str,
        default="match_results.txt",
        help="path to append-only match result output",
    )
    args = parser.parse_args()

    if args.games_per_color <= 0:
        raise ValueError("games_per_color must be positive.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RulesConfig(board_size=args.board_size, win_k=args.win_k, hp_max=args.hp_max)

    if not os.path.isdir(args.savePath):
        raise FileNotFoundError(f"savePath not found: {args.savePath}")

    model_paths = _list_model_files_by_mtime(args.savePath)
    if len(model_paths) == 0:
        raise RuntimeError("No model files found to evaluate.")

    print(f"[test-elo] found {len(model_paths)} model(s) in {args.savePath}")
    print(f"[test-elo] each pairing plays {2 * args.games_per_color} games")
    match_results = evaluate_all_models(
        model_paths=model_paths,
        cfg=cfg,
        games_per_color=int(args.games_per_color),
        sims=int(args.sim),
        device=device,
        output_path=args.output,
    )
    for result in match_results:
        print(
            f"{os.path.basename(result.path_a)}\t"
            f"{os.path.basename(result.path_b)}\t"
            f"{result.winner_name}"
        )
    print(f"[test-elo] wrote {args.output}")


if __name__ == "__main__":
    main()
