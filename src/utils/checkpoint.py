from __future__ import annotations
import os
import glob
import torch
from typing import Optional


def ensure_dir(path: str) -> None:
    """若目录不存在则创建。"""
    os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    **extra) -> None:
    """保存模型与优化器状态；extra 可携带 global_step 等信息。"""
    ensure_dir(os.path.dirname(path) or ".")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                    map_location: Optional[str] = None) -> dict:
    """加载检查点到给定模型/优化器；返回 payload 以便读取自定义字段。"""
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def latest_checkpoint_path(save_dir: str, pattern: str = "ckpt_*.pt") -> Optional[str]:
    """返回目录中“最新”的检查点路径；按文件修改时间排序。无则返回 None。"""
    ensure_dir(save_dir)
    candidates = glob.glob(os.path.join(save_dir, pattern))
    if not candidates:
        return None
    # 以 mtime 降序，取最新
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]