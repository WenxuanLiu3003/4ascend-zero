"""Selectable CPU inference shared by self-play, evaluation and the UI.

PyTorch remains the training/checkpoint format. Export is lazy and cached by
weight content, architecture and exporter version; no training jobs are started.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Union


def inference_backend():
    backend = os.environ.get("ASCEND_INFERENCE_BACKEND", "onnx").strip().lower()
    if backend not in ("onnx", "pytorch"):
        raise ValueError("ASCEND_INFERENCE_BACKEND must be 'onnx' or 'pytorch'")
    return backend


class PyTorchPolicyValue:
    backend = "pytorch"

    def __init__(self, model):
        import torch
        from .model import PolicyValueNet
        # Never call eval()/cpu() on the caller's training model.
        with torch.random.fork_rng(devices=[]):
            self.model = PolicyValueNet(in_planes=model.stem[0].in_channels,
                                        board_size=model.board_size,
                                        width=model.stem[0].out_channels,
                                        n_blocks=len(model.res)).cpu().float().eval()
        self.model.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()}, strict=True)

    def eval(self):
        return self

    def close(self):
        self.model = None

    def __call__(self, x):
        import torch
        if self.model is None:
            raise RuntimeError("PyTorch inference model has been released")
        if x.device.type != "cpu" or x.dtype != torch.float32:
            raise ValueError("PyTorch inference requires a CPU float32 tensor")
        with torch.inference_mode():
            return self.model(x)


def inference_threads(workers: int = 1) -> int:
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    available = min(available, int(os.environ.get("SLURM_CPUS_PER_TASK", available)))
    requested = int(os.environ.get("ASCEND_ONNX_THREADS", min(8, available)))
    if requested < 1 or workers < 1 or available < workers:
        raise ValueError("Invalid ONNX thread budget or more workers than allocated CPUs")
    return max(1, min(requested, available) // workers)


class OnnxPolicyValue:
    backend = "onnx"

    def __init__(self, path, threads=None):
        import onnxruntime as ort
        self.path = str(path)
        self.threads = inference_threads() if threads is None else int(threads)
        if self.threads < 1:
            raise ValueError("ONNX threads must be positive")
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(self.path, sess_options=options, providers=["CPUExecutionProvider"])
        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("CPUExecutionProvider required")

    def eval(self):
        return self

    def close(self):
        self.session = None

    def __call__(self, x):
        import torch
        if self.session is None:
            raise RuntimeError("ONNX inference session has been released")
        if x.device.type != "cpu" or x.dtype != torch.float32:
            raise ValueError("ONNX inference requires a CPU float32 tensor")
        p, v = self.session.run(["policy_logits", "value"], {"state": x.detach().numpy()})
        return torch.from_numpy(p), torch.from_numpy(v)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validation_inputs(config):
    """Use actual encoded positions; preserve game RNG around the export check."""
    import random
    import numpy as np
    import torch
    from ..core.board import Board
    from ..core.encoding import AlphaZeroStateEncoder
    from ..core.engine import Engine
    from ..core.rules import RulesConfig
    from ..core.state import GameState
    cfg = RulesConfig(board_size=config["board_size"])
    encoder, engine = AlphaZeroStateEncoder(last_k=8), Engine(win_k=cfg.win_k)
    if config["in_planes"] != encoder.num_planes:
        raise ValueError("Model input planes do not match the game encoder")
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        random.seed(42)
        np.random.seed(42)
        state = GameState(cfg=cfg, board=Board(cfg.board_size))
        inputs = []
        for _ in range(32):
            inputs.append(torch.from_numpy(encoder.encode(state, state.to_play)).unsqueeze(0))
            if state.is_terminal():
                state = GameState(cfg=cfg, board=Board(cfg.board_size))
            state = engine.step(state, random.choice(state.legal_moves()))
        return inputs
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def ensure_onnx(model, threads=None):
    """Create a detached CPU snapshot; never change training mode/device/weights."""
    if isinstance(model, OnnxPolicyValue):
        return model
    import torch
    import onnx
    import numpy as np
    from filelock import FileLock
    from .model import PolicyValueNet

    state = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
    config = dict(in_planes=model.stem[0].in_channels, board_size=model.board_size,
                  width=model.stem[0].out_channels, n_blocks=len(model.res))
    signature = dict(config=config, torch=torch.__version__, onnx=onnx.__version__, opset=17,
                     architecture=_digest(Path(__file__).with_name("model.py")), exporter=_digest(__file__))
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode())
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    key = digest.hexdigest()
    cache = Path(os.environ.get("ASCEND_ONNX_CACHE_DIR", Path(__file__).resolve().parents[2] / ".onnx_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    path, manifest = cache / f"{key}.onnx", cache / f"{key}.json"
    with FileLock(str(cache / f"{key}.lock"), timeout=600):
        valid = False
        if path.is_file() and manifest.is_file():
            try:
                valid = json.loads(manifest.read_text())["onnx_sha256"] == _digest(path)
            except (KeyError, ValueError, OSError):
                pass
        if not valid:
            # Snapshot construction and tracing must not consume training RNG.
            previous_threads = torch.get_num_threads()
            try:
                torch.set_num_threads(1)
                with torch.random.fork_rng(devices=[]):
                    snapshot = PolicyValueNet(**config).cpu().float().eval()
                    snapshot.load_state_dict(state, strict=True)
                    shape = (1, config["in_planes"], config["board_size"], config["board_size"])
                    with tempfile.TemporaryDirectory(prefix=f".{key}-", dir=cache) as temp:
                        candidate = Path(temp) / "model.onnx"
                        with torch.inference_mode():
                            torch.onnx.export(snapshot, torch.zeros(shape), str(candidate), dynamo=False,
                                              opset_version=17, input_names=["state"],
                                              output_names=["policy_logits", "value"])
                        onnx.checker.check_model(str(candidate))
                        runner = OnnxPolicyValue(candidate, threads=1)
                        errors = [0.0, 0.0]
                        with torch.inference_mode():
                            for x in _validation_inputs(config):
                                expected, actual = snapshot(x), runner(x)
                                for j in range(2):
                                    np.testing.assert_allclose(actual[j].numpy(), expected[j].numpy(), rtol=1e-4, atol=1e-5)
                                    errors[j] = max(errors[j], float((actual[j] - expected[j]).abs().max()))
                        del runner
                        info = Path(temp) / "manifest.json"
                        info.write_text(json.dumps(dict(signature=signature, max_abs_error=errors,
                                                        onnx_sha256=_digest(candidate)), indent=2))
                        os.replace(candidate, path)
                        os.replace(info, manifest)
                        print(f"[onnx] Exported and validated CPU FP32 model: {path}", flush=True)
            finally:
                torch.set_num_threads(previous_threads)
    return OnnxPolicyValue(path, threads=threads)


InferenceModel = Union[OnnxPolicyValue, PyTorchPolicyValue]


def ensure_inference(model, threads=None, backend=None):
    backend = inference_backend() if backend is None else backend
    if backend not in ("onnx", "pytorch"):
        raise ValueError(f"Unsupported inference backend: {backend}")
    if isinstance(model, (OnnxPolicyValue, PyTorchPolicyValue)):
        if model.backend != backend:
            raise ValueError("Backend changed after model creation; reload the checkpoint with the new backend")
        return model
    return ensure_onnx(model, threads=threads) if backend == "onnx" else PyTorchPolicyValue(model)


def load_inference_checkpoint(path=None, board_size=9, threads=None, backend=None):
    backend = inference_backend() if backend is None else backend
    import torch
    from .model import PolicyValueNet
    # Preserve the caller's RNG when constructing a model to load saved weights.
    if path is None:
        model = PolicyValueNet(in_planes=25, board_size=board_size)
    else:
        with torch.random.fork_rng(devices=[]):
            model = PolicyValueNet(in_planes=25, board_size=board_size)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        del payload
        print(f"[{backend}] CPU checkpoint: {path}", flush=True)
    return ensure_inference(model, threads=threads, backend=backend)


def load_onnx_checkpoint(path=None, board_size=9, threads=None):
    """Explicit ONNX loader retained for conversion checks and existing callers."""
    return load_inference_checkpoint(path, board_size, threads, backend="onnx")
