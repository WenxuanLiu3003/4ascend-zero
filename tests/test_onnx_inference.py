"""Run model tests on a Slurm compute node, never on the login node."""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("SLURM_JOB_ID"), reason="Model tests require a Slurm compute allocation")


@pytest.fixture(autouse=True)
def default_backend(monkeypatch, tmp_path):
    monkeypatch.delenv("ASCEND_INFERENCE_BACKEND", raising=False)
    yield
    # Test samples must never linger or mix with production datasets.
    for path in tmp_path.rglob("*dataset*.pt"):
        path.unlink()


def test_export_snapshot_cache_and_training(tmp_path, monkeypatch):
    import torch
    import numpy as np
    from src.ai.model import PolicyValueNet
    from src.ai.inference import ensure_onnx

    torch.set_num_threads(1)
    monkeypatch.setenv("ASCEND_ONNX_CACHE_DIR", str(tmp_path))
    model = PolicyValueNet(25, width=4, n_blocks=1).train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    runner = ensure_onnx(model, threads=1)
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    stamp = Path(runner.path).stat().st_mtime_ns
    again = ensure_onnx(model, threads=1)
    assert runner.path == again.path and Path(runner.path).stat().st_mtime_ns == stamp
    model.eval()
    x = torch.randn(1, 25, 9, 9)
    with torch.inference_mode():
        for expected, actual in zip(model(x), runner(x)):
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-5)
    with torch.no_grad():
        model.p_fc.bias.add_(0.01)
    updated = ensure_onnx(model, threads=1)
    assert updated.path != runner.path
    # An interrupted/corrupted cache cannot be accepted as valid.
    Path(updated.path).write_bytes(b"invalid")
    repaired = ensure_onnx(model, threads=1)
    repaired(x)


def test_selfplay_dataset_training_and_all_entrypoints(tmp_path, monkeypatch):
    import torch
    import numpy as np
    from src.train import AZLiteTrainer
    from src.ai.inference import ensure_onnx, load_onnx_checkpoint, OnnxPolicyValue
    from src.test import play_one_game as evaluate
    from src.test_elo import play_one_game as evaluate_elo, _load_model
    from src.replay.selfplay_trace import generate_selfplay_trace
    from src.ui.pygame_app import MCTSRunner
    from src.core.engine import Engine
    from src.core.state import GameState
    from src.core.board import Board

    torch.set_num_threads(1)
    monkeypatch.setenv("ASCEND_ONNX_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ASCEND_ONNX_THREADS", "1")
    trainer = AZLiteTrainer(save_dir=str(tmp_path / "checkpoints"), device="cpu")
    # Constructor/gradient-only path must not export anything.
    assert not (tmp_path / "cache").exists()
    trainer.cfg.max_turns = 2  # Functional test only; production remains 225.
    data = trainer.self_play_batch(games=1, sims=2)
    assert len(data) == 16 and all(len(sample) == 5 for sample in data)
    path = tmp_path / "dataset.pt"
    torch.save(data, path)
    loaded = torch.load(path, weights_only=False)
    old_runner = ensure_onnx(trainer.model)
    old_weights = trainer.model.p_fc.weight.detach().clone()
    losses = trainer.train_step(loaded)
    assert np.isfinite(losses).all()
    assert not torch.equal(old_weights, trainer.model.p_fc.weight)
    runner = ensure_onnx(trainer.model)
    assert runner.path != old_runner.path and trainer.model.training
    checkpoint = tmp_path / "updated.pt"
    torch.save({"model": trainer.model.state_dict()}, checkpoint)
    assert load_onnx_checkpoint(checkpoint).path == runner.path
    assert isinstance(_load_model(str(checkpoint), trainer.cfg, "cuda"), OnnxPolicyValue)
    evaluate(runner, runner, trainer.cfg, sims=2, device="cuda")
    evaluate_elo(runner, runner, trainer.cfg, sims=2, device="cuda")
    trace = generate_selfplay_trace(trainer.cfg, runner, device="cuda", sims=2)
    assert len(trace) >= 2
    monkeypatch.setattr("src.ui.pygame_app.latest_checkpoint_path", lambda _: str(checkpoint))
    gui = MCTSRunner(trainer.cfg, Engine())
    gui.ensure_loaded()
    assert gui.device == "cpu" and isinstance(gui.model, OnnxPolicyValue)
    gui.mcts.sims = 2
    assert gui.best_moves(GameState(cfg=trainer.cfg, board=Board(9)))
    # Exercise the process worker payload without crossing live ORT sessions.
    from dataclasses import asdict
    worker_data = trainer._worker_self_play((runner.path, 1, asdict(trainer.cfg), 2, False))
    assert len(worker_data) == 16


@pytest.mark.parametrize("backend", ["onnx", "pytorch"])
def test_backend_switch_does_not_change_training(tmp_path, monkeypatch, backend):
    import copy
    import importlib.abc
    import sys
    import torch
    import numpy as np
    from src.train import AZLiteTrainer
    from src.ai.inference import ensure_inference, load_inference_checkpoint, inference_backend
    from src.test_elo import _load_model
    from src.test import play_one_game
    from src.ui.pygame_app import MCTSRunner

    class BlockOnnxImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in ("onnx", "onnxruntime"):
                raise AssertionError("PyTorch-only inference must not import ONNX")

    torch.set_num_threads(1)
    monkeypatch.setenv("ASCEND_ONNX_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ASCEND_ONNX_THREADS", "1")
    # Even an invalid inference setting must not affect gradient-only setup/update.
    monkeypatch.setenv("ASCEND_INFERENCE_BACKEND", "not-a-backend")
    trainer = AZLiteTrainer(save_dir=str(tmp_path / "checkpoints"), device="cpu")
    batch = [(np.zeros((25,9,9),np.float32), np.full(81,1/81,np.float32), 1, 0.0, 0.0)] * 2
    trainer.train_step(batch)  # Initialize Adam state before testing preservation.
    assert not (tmp_path / "cache").exists()
    reference = copy.deepcopy(trainer)
    monkeypatch.setenv("ASCEND_INFERENCE_BACKEND", backend)
    assert inference_backend() == backend
    if backend == "pytorch":
        # Block imports even if the preceding ONNX test loaded the packages.
        monkeypatch.setattr(sys, "meta_path", [BlockOnnxImports()] + sys.meta_path)
        for key in list(sys.modules):
            if key.split('.')[0] in ("onnx", "onnxruntime"):
                monkeypatch.delitem(sys.modules, key)
    rng = torch.get_rng_state().clone()
    original_threads = torch.get_num_threads()
    runner = ensure_inference(trainer.model)
    assert runner.backend == backend and trainer.model.training
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.get_num_threads() == original_threads
    runner(torch.zeros(1,25,9,9))
    # Same data produces the exact same gradient update and optimizer state.
    assert trainer.train_step(batch) == reference.train_step(batch)
    for k, v in trainer.model.state_dict().items():
        assert torch.equal(v, reference.model.state_dict()[k])
    for key, values in trainer.opt.state_dict()['state'].items():
        for field, value in values.items():
            other = reference.opt.state_dict()['state'][key][field]
            assert torch.equal(value, other) if torch.is_tensor(value) else value == other
    trainer.cfg.max_turns = 2
    assert len(trainer.self_play_batch(games=1, sims=2)) == 16
    checkpoint = tmp_path/'checkpoint.pt'
    torch.save({'model': trainer.model.state_dict()}, checkpoint)
    loaded = load_inference_checkpoint(checkpoint)
    assert loaded.backend == backend
    assert _load_model(str(checkpoint), trainer.cfg, 'cpu').backend == backend
    play_one_game(loaded, loaded, trainer.cfg, sims=2, device='cpu')
    monkeypatch.setattr('src.ui.pygame_app.latest_checkpoint_path', lambda _: str(checkpoint))
    gui = MCTSRunner(trainer.cfg, trainer.engine)
    gui.ensure_loaded()
    assert gui.model.backend == backend
    from src.replay.selfplay_trace import generate_selfplay_trace
    assert len(generate_selfplay_trace(trainer.cfg, loaded, sims=2)) >= 2
    from dataclasses import asdict
    source = loaded.path if backend == 'onnx' else trainer.model.state_dict()
    assert len(trainer._worker_self_play((source, 1, asdict(trainer.cfg), 2, False))) == 16
    if backend == 'pytorch':
        assert not (tmp_path / 'cache').exists()


def test_gui_cuda_is_isolated(tmp_path, monkeypatch):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("GPU allocation required for the GUI CUDA test")
    from src.ai.model import PolicyValueNet
    from src.core.rules import RulesConfig
    from src.core.engine import Engine
    from src.core.board import Board
    from src.core.state import GameState
    from src.ui.pygame_app import MCTSRunner

    checkpoint = tmp_path / "gui.pt"
    torch.save({"model": PolicyValueNet(25).state_dict()}, checkpoint)
    monkeypatch.setattr("src.ui.pygame_app.latest_checkpoint_path", lambda _: str(checkpoint))
    monkeypatch.setenv("ASCEND_INFERENCE_BACKEND", "onnx")
    def disallow_cpu_adapter(*a, **kw):
        raise AssertionError("GUI CUDA must bypass CPU adapters and ONNX export")
    monkeypatch.setattr("src.ai.inference.ensure_inference", disallow_cpu_adapter)
    cfg = RulesConfig(max_turns=2)
    runner = MCTSRunner(cfg, Engine(), device="cuda")
    runner.ensure_loaded()
    assert runner.mcts.device == "cuda" and runner.mcts.sims == 800
    assert next(runner.model.parameters()).device.type == "cuda"
    assert not runner.model.training
    runner.mcts.sims = 2
    assert runner.best_moves(GameState(cfg=cfg, board=Board(9)))


def test_gui_missing_cuda_does_not_fall_back(monkeypatch):
    import torch
    from src.core.rules import RulesConfig
    from src.core.engine import Engine
    from src.ui.pygame_app import MCTSRunner
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("src.ui.pygame_app.latest_checkpoint_path", lambda _: None)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        MCTSRunner(RulesConfig(), Engine(), device="cuda").ensure_loaded()
