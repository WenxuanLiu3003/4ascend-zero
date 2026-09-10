from __future__ import annotations
import torch

from ..utils.checkpoint import latest_checkpoint_path
from ..core.rules import RulesConfig
from ..ai.inference import load_inference_checkpoint
from .selfplay_trace import generate_selfplay_trace
from ..ui.replay_viewer import launch_replay


def main():
    device = "cpu"
    torch.set_num_threads(1)

    # 规则与模型尺寸
    cfg = RulesConfig(board_size=9, win_k=4, hp_max=6)

    # 加载最新 checkpoint（若存在）
    ckpt = latest_checkpoint_path("checkpoints")
    if ckpt is not None:
        print(f"[replay] Load latest checkpoint: {ckpt}")
    else:
        print("[replay] No checkpoint found. Using randomly initialized model.")

    model = load_inference_checkpoint(ckpt, board_size=cfg.board_size)

    # 生成一局完整自对弈的轨迹（不复用树，避免植物随机带来的偏差）
    trace = generate_selfplay_trace(cfg, model, device=device, sims=1600, temp_steps=0,
                                    dir_alpha=0.3, dir_eps=0.25, use_tree_reuse=False)

    # 启动回放 GUI
    launch_replay(trace, board_size=cfg.board_size)


if __name__ == "__main__":
    main()
