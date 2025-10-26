from __future__ import annotations
from typing import List
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import multiprocessing as mp
from .core.types import Player

from .core.rules import RulesConfig
from .core.board import Board
from .core.state import GameState
from .core.engine import Engine
from .core.encoding import AlphaZeroStateEncoder
from .ai.model import PolicyValueNet
from .ai.selfplay import SelfPlay
import argparse
from .utils.checkpoint import (
    ensure_dir, latest_checkpoint_path, save_checkpoint, load_checkpoint,
)

class AZLiteTrainer:
    def __init__(self, board_size=9, win_k=4, hp_max=6, device="cpu",
                 save_dir: str = "checkpoints", save_every_sec: int = 300,
                 num_workers: int = 0,
                 shaped_reward_enabled: bool = False,
                 shaped_reward_coeff: float = 0.05,
                 reuse_tree: bool = False):
        self.cfg = RulesConfig(board_size=board_size, win_k=win_k, hp_max=hp_max)
        self.engine = Engine(win_k=self.cfg.win_k)
        self.encoder = AlphaZeroStateEncoder(last_k=8)
        self.device = device

        self.model = PolicyValueNet(in_planes=self.encoder.num_planes, board_size=board_size).to(device)
        print("trainable parameters: %d" % sum(p.numel() for p in self.model.parameters() if p.requires_grad) )
        self.model.eval()
        self.opt = optim.Adam(self.model.parameters(), lr=0.02, weight_decay=1e-4)
        self.ce = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()

        self.save_dir = save_dir
        self.save_every_sec = save_every_sec
        ensure_dir(self.save_dir)
        self.global_step = 0
        self.num_workers = max(0, int(num_workers))

        # 奖励塑形开关与系数
        self.shaped_reward_enabled = bool(shaped_reward_enabled)
        self.shaped_reward_coeff = float(shaped_reward_coeff)

        # 自博弈时是否启用根节点树复用（你的棋有随机植物，默认关）
        self.reuse_tree = bool(reuse_tree)

        ckpt = latest_checkpoint_path(self.save_dir)
        if ckpt is not None:
            print(f"[train] Loading latest checkpoint: {ckpt}")
            payload = load_checkpoint(ckpt, self.model, self.opt, map_location=device)
            self.global_step = int(payload.get("global_step", self.global_step))
        else:
            print("[train] No checkpoint found. Starting fresh.")

    # —— 单进程自博弈 —— #
    def _self_play_batch_serial(self, games=8, sims=400) -> List[Tuple[np.ndarray, np.ndarray, int, float]]:
        sp = SelfPlay(self.model, self.encoder, self.engine,
                      board_size=self.cfg.board_size, sims=sims, device=self.device,
                      use_tree_reuse=self.reuse_tree)
        dataset = []
        for _ in tqdm(range(games), desc="Self-play games", unit="game"):
            s = GameState(cfg=self.cfg, board=Board(self.cfg.board_size), to_play=Player.BLACK)
            data = sp.play_one(s)  # 返回 (planes, pi, z, aux_r)
            for sample in data:
                dataset.extend(SelfPlay.augment(sample))  # 也返回四元组
        random.shuffle(dataset)
        print("length of dataset: %d" % len(dataset))
        return dataset

    # —— 多进程 worker —— #
    @staticmethod
    def _worker_self_play(payload):
        state_dict, cfg_dict, sims, reuse_tree = payload
        encoder = AlphaZeroStateEncoder(last_k=8)
        engine = Engine(win_k=cfg_dict['win_k'])
        model = PolicyValueNet(in_planes=encoder.num_planes, board_size=cfg_dict['board_size']).to('cpu')
        model.load_state_dict(state_dict)
        model.eval()
        sp = SelfPlay(model, encoder, engine, board_size=cfg_dict['board_size'],
                      sims=sims, device='cpu', use_tree_reuse=reuse_tree)
        s = GameState(cfg=RulesConfig(**cfg_dict), board=Board(cfg_dict['board_size']))
        data = sp.play_one(s)  # (planes, pi, z, aux_r)
        out = []
        for sample in data:
            out.extend(SelfPlay.augment(sample))
        return out

    def self_play_batch(self, games=8, sims=400) -> List[Tuple[np.ndarray, np.ndarray, int, float]]:
        if self.num_workers <= 0:
            return self._self_play_batch_serial(games=games, sims=sims)

        # 广播参数到子进程
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        cfg_dict = {
            'board_size': self.cfg.board_size,
            'win_k': self.cfg.win_k,
            'hp_max': self.cfg.hp_max,
            'allow_overlap_plants': self.cfg.allow_overlap_plants,
            'max_turns': self.cfg.max_turns,
            'ad_attacker_hp_delta_on_fail': self.cfg.ad_attacker_hp_delta_on_fail,
            'ad_defender_hp_delta_on_fail': self.cfg.ad_defender_hp_delta_on_fail,
            'ad_attacker_hp_delta_on_success': self.cfg.ad_attacker_hp_delta_on_success,
            'ad_defender_hp_delta_on_success': self.cfg.ad_defender_hp_delta_on_success,
        }
        tasks = [(state_dict, cfg_dict, sims, self.reuse_tree) for _ in range(games)]
        dataset = []
        with mp.get_context("spawn").Pool(processes=self.num_workers) as pool:
            for out in tqdm(pool.imap_unordered(self._worker_self_play, tasks),
                            total=games, desc="Self-play (mp)", unit="game"):
                dataset.extend(out)
        random.shuffle(dataset)
        return dataset

    def train_step(self, batch: List[Tuple[np.ndarray, np.ndarray, int, float]]):
        xs, target_pi, target_v, aux = [], [], [], []
        for x, pi, z, a_r in batch:
            xs.append(x)
            target_pi.append(pi)
            target_v.append(z)
            aux.append(a_r)

        x = torch.from_numpy(np.stack(xs)).float().to(self.device)                # [B,C,H,W]
        pi = torch.from_numpy(np.stack(target_pi)).float().to(self.device)        # [B,81]
        z = torch.from_numpy(np.array(target_v, dtype=np.float32)).to(self.device) # [B]
        aux_r = torch.from_numpy(np.array(aux, dtype=np.float32)).to(self.device)  # [B]

        # 奖励塑形：z' = clip(z + λ * aux_r, -1, 1)（开关控制）
        if self.shaped_reward_enabled:
            z_target = torch.clamp(z + self.shaped_reward_coeff * aux_r, -1.0, 1.0)
        else:
            z_target = z

        self.model.train()
        p_logits, v = self.model(x)  # p_logits:[B,81], v:[B]
        logp = torch.log_softmax(p_logits, dim=-1)

        loss_p = self.ce(logp, pi)       # 策略 KL
        loss_v = self.mse(v, z_target)   # 价值 MSE（可选加形）
        loss = loss_p + loss_v

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        self.global_step += 1
        return float(loss.item()), float(loss_p.item()), float(loss_v.item())

    def self_play_and_train_epoch(self, games_per_epoch: int, sims: int,
                                  batch_size: int) -> None:
        data = self.self_play_batch(games=games_per_epoch, sims=sims)
        print(f"[train] Collected samples: {len(data)}")
        for i in tqdm(range(0, len(data), batch_size), desc="Train", unit="batch"):
            batch = data[i:i+batch_size]
            loss, lp, lv = self.train_step(batch)
            tqdm.write(f"[train] step={self.global_step} loss={loss:.4f} (p={lp:.4f}, v={lv:.4f})")

    def train_loop(self, epochs: int = 1, games_per_epoch: int = 8, sims: int = 400,
                   batch_size: int = 256):
        last_save_t = time.time()
        try:
            for ep in range(epochs):
                print(f"[train] Epoch {ep+1}/{epochs}: self-play generating...")
                self.self_play_and_train_epoch(games_per_epoch, sims, batch_size)

                # TODO: modify the training method by selfPlay with last epoch. If win > 50% then update the parameters, otherwise do not update.

                # now = time.time()
                # if now - last_save_t >= self.save_every_sec:
                #     path = os.path.join(self.save_dir, f"ckpt_step{self.global_step}.pt")
                #     save_checkpoint(path, self.model, self.opt, global_step=self.global_step)
                #     print(f"[train] checkpoint saved: {path}")
                #     last_save_t = now

                # 每个 epoch 也落一个
                # if ep % 10 == 9:
                path = os.path.join(self.save_dir, f"ckpt_ep{ep+1}_step{self.global_step}.pt")
                save_checkpoint(path, self.model, self.opt, global_step=self.global_step)
                print(f"[train] checkpoint saved (epoch end): {path}")
        except KeyboardInterrupt:
            path = os.path.join(self.save_dir, f"ckpt_interrupt_step{self.global_step}.pt")
            save_checkpoint(path, self.model, self.opt, global_step=self.global_step)
            print(f"[train] checkpoint saved (interrupt): {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training parameters")
    parser.add_argument('--epoch', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--sim', type=int, default=1000, help='Number of simulations')
    parser.add_argument('--savePath', type=str, default="checkpoints", help='model path')
    parser.add_argument('--game', type=int, default=500, help='Number of games per epoch')
    parser.add_argument('--batch', type=int, default=256, help='Number of batch')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = AZLiteTrainer(board_size=9, win_k=4, hp_max=6, device=device,
                            save_dir=args.savePath, save_every_sec=300,
                            num_workers=0,
                            shaped_reward_enabled=False,   # ← 打开/关闭 奖励塑形
                            shaped_reward_coeff=0.05,     # ← 微奖励系数 λ
                            reuse_tree=False)             # ← 是否根复用（默认关闭）
    trainer.train_loop(epochs=args.epoch, games_per_epoch=args.game, sims=args.sim, batch_size=args.batch)