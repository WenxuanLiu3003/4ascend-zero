from __future__ import annotations

import argparse
import glob
import os
from typing import Optional

import torch

path = "/insomnia001/depts/free/users/wl3003/4ascend-model/checkpoints"
out_path = "/insomnia001/depts/free/users/wl3003/4ascend-model/model-v3/checkpoints"


def latest_checkpoint_path(save_dir: str, pattern: str = "ckpt_*.pt") -> Optional[str]:
    candidates = glob.glob(os.path.join(save_dir, pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _expand_stem_weight_22_to_25(weight_22: torch.Tensor) -> torch.Tensor:
    if weight_22.ndim != 4:
        raise ValueError(f"stem.0.weight must be 4D, got shape={tuple(weight_22.shape)}")
    out_ch, in_ch, kh, kw = weight_22.shape
    if in_ch != 22:
        raise ValueError(f"expected 22 input channels, got {in_ch}")

    # New channel layout:
    # 0..7   <- old 0..7
    # 8..10  <- zeros (new channels)
    # 11..24 <- old 8..21
    weight_25 = weight_22.new_zeros((out_ch, 25, kh, kw))
    weight_25[:, :8, :, :] = weight_22[:, :8, :, :]
    weight_25[:, 11:, :, :] = weight_22[:, 8:, :, :]
    return weight_25


def convert_latest_checkpoint(save_dir: str, output_dir: str) -> Optional[str]:
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"checkpoint directory not found: {save_dir}")
    os.makedirs(output_dir, exist_ok=True)

    ckpt = latest_checkpoint_path(save_dir, pattern="ckpt_*.pt")
    if ckpt is None:
        print(f"[adjust] no checkpoint files found under: {save_dir}")
        return None

    print(f"[adjust] latest checkpoint: {ckpt}")
    payload = torch.load(ckpt, map_location="cpu")
    if "model" not in payload or not isinstance(payload["model"], dict):
        raise ValueError(f"invalid checkpoint payload format: {ckpt}")

    model_state = payload["model"]
    stem_key = "stem.0.weight"
    if stem_key not in model_state:
        raise KeyError(f"{stem_key} not found in checkpoint model state")

    stem_weight = model_state[stem_key]
    in_ch = int(stem_weight.shape[1])

    if in_ch == 25:
        print("[adjust] latest checkpoint already uses 25 input channels. no conversion.")
        return ckpt
    if in_ch != 22:
        raise ValueError(f"[adjust] unsupported input channels in stem: {in_ch} (expect 22 or 25)")

    model_state[stem_key] = _expand_stem_weight_22_to_25(stem_weight)
    payload["model"] = model_state

    # Old optimizer states contain 22-channel tensors for stem.0.weight.
    # Remove them to avoid shape mismatch during resumed training.
    payload.pop("optimizer", None)
    payload["converted_from_in_channels"] = 22
    payload["converted_to_in_channels"] = 25
    payload["converted_from_checkpoint"] = os.path.basename(ckpt)

    base = os.path.basename(ckpt)
    stem, ext = os.path.splitext(base)
    out_base = f"{stem}_in25{ext or '.pt'}"
    output_path = os.path.join(output_dir, out_base)
    suffix = 1
    while os.path.exists(output_path):
        output_path = os.path.join(output_dir, f"{stem}_in25_{suffix}{ext or '.pt'}")
        suffix += 1

    torch.save(payload, output_path)
    print(f"[adjust] converted checkpoint saved: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert latest 22-channel checkpoint to 25 channels.")
    parser.add_argument("--path", type=str, default=path, help="Input directory containing ckpt_*.pt files.")
    parser.add_argument("--out_path", type=str, default=out_path, help="Output directory for converted checkpoint.")
    args = parser.parse_args()
    convert_latest_checkpoint(args.path, args.out_path)


if __name__ == "__main__":
    main()
