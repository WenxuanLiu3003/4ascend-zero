# 4ascend-Zero: Alphago-Zero-Style AI for the 4ascend Game

## CPU ONNX inference / GPU PyTorch training

All MCTS inference (self-play, match/Elo evaluation, replay generation and the
pygame UI) defaults to **ONNX Runtime on CPU, FP32**, with optional PyTorch CPU
inference. Gradient training remains
PyTorch, using CUDA when available. Dataset `.pt` files and training checkpoints
keep their existing formats.

Install inference dependencies into the existing virtual environment:

```bash
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
```

PyTorch 2.6 or newer is needed for the explicit exporter API (the current HPC
environment uses 2.8). No ncnn/pnnx dependencies are required for production.

The manual HPC workflow stays the same; submit from the project root:

```bash
sbatch play.sh       # generate data: CPU ONNX, sim=2400, 80 games per task/run
sbatch train.sh      # run later when enough data exists: original GPU training
sbatch test.sh       # CPU ONNX match evaluation
sbatch test-elo.sh   # CPU ONNX Elo evaluation
```

Switch inference backend for a new process/job with `ASCEND_INFERENCE_BACKEND`:

```bash
ASCEND_INFERENCE_BACKEND=onnx sbatch play.sh     # default
ASCEND_INFERENCE_BACKEND=pytorch sbatch play.sh
ASCEND_INFERENCE_BACKEND=pytorch sbatch test.sh
ASCEND_INFERENCE_BACKEND=pytorch sbatch test-elo.sh
ASCEND_INFERENCE_BACKEND=pytorch python -m src.ui.pygame_app
```

The same setting covers replay self-play. Both backends stay on CPU; this environment
variable does not enable GPU inference. Existing running jobs keep their backend. Invalid
values fail explicitly when inference is requested. `--trainOnly` ignores this
variable completely: its model, training device, gradient updates and optimizer
are unaffected. `train.sh` is unchanged. Inference uses a detached CPU snapshot,
so it never changes the caller's training model mode, device or parameters.
PyTorch inference needs no ONNX packages and creates no ONNX cache. CLI/UI
PyTorch inference uses one Torch thread, as in the earlier CPU measurements;
`ASCEND_ONNX_THREADS` applies only to ONNX inference.

For **pygame human-vs-AI only**, restore PyTorch CUDA inference explicitly:

```bash
python -m src.ui.pygame_app --device cuda
```

This loads the same `.pt` checkpoint directly into PyTorch on the GPU, bypassing
ONNX and `ASCEND_INFERENCE_BACKEND`. CUDA must be available; otherwise the app
reports an error instead of silently changing backend. This mode restores the
original GPU search default of 800 simulations. `--device cpu` (or no flag)
keeps 400 simulations and the CPU backend selected above. This option does not
affect batch self-play, replay generation, test/Elo scripts or gradient training.

`play.sh`, `test.sh` and `test-elo.sh` request **8 CPUs and 2GB RAM per array task**,
with no GPU allocation. Their original array sizes and play requeue policy are
retained; failures stop instead of triggering another requeue. No new scheduling,
automatic training, checkpoint watching or dataset deletion is added.

Each process loads its selected `.pt` checkpoint at startup. In ONNX mode it
automatically exports a validated ONNX model if needed. New weights produce a different cache
entry on the next run. An already running job keeps its loaded weights. Models
are cached in `.onnx_cache/`, with locks and atomic publication for concurrent
array jobs. `ASCEND_ONNX_CACHE_DIR` can override the cache directory, and
`ASCEND_ONNX_THREADS` overrides the inference thread budget (default up to 8).
Multiprocess self-play divides that budget among workers. Export failures or
numerical validation failures are reported; there is no silent backend fallback.

The existing `ASCEND_CHECKPOINT_DIR` / `--savePath` behavior is retained for the
CLI scripts. GUI/replay still use their existing `checkpoints/` location. The
pygame CPU search default is 400 simulations. Search, rules, temperatures,
Dirichlet noise and augmentation are otherwise unchanged. Self-play uses 2400
simulations by default, increased to 3600 when more than 50 stones occupy the
board. Each task/run plays 80 games, saving one dataset file per ten-game chunk.

`--trainOnly` does not load ONNX or create inference sessions. If the existing
combined self-play/training mode is used, each self-play batch snapshots the
current training weights on CPU without changing the training model or optimizer.

Model integration tests must run on a compute node:
`python -m pytest -q tests/test_onnx_inference.py`.
They skip outside a Slurm allocation to avoid inference on login nodes. They
cover output consistency, cache reuse/corruption, dataset compatibility, both
backends and the inference entrypoints. They also verify identical gradient and
optimizer updates after inference, and no ONNX imports in PyTorch mode.
These short tests do not establish the
peak RAM requirement of a complete ten-game production batch; monitor `MaxRSS`
and override `sbatch --mem=...` if the production workload needs more than 2GB.

The old `selfplay_benchmark/` experiment directory has been removed. Original
copies of the three previously Git-ignored shell scripts are retained locally in
`.onnx_cache/script-backups/` for rollback; the new versions are no longer ignored.

## Overview
**4ascend-Zero** is an AlphaZero-style reinforcement learning system built for the **4ascend board game**, a 9×9 strategy game featuring dynamic attack–defense mechanics, plant-based resource tiles, and HP-based victory conditions. (see [4ASCEND | フリーゲーム投稿サイト unityroom](https://unityroom.com/games/4ascend))

The project combines:
- **Monte Carlo Tree Search (MCTS)**
- **Deep Residual Policy–Value Networks**
- **Self-play Reinforcement Learning**

---
## How to Deploy and Play with My Own AI model?

Step 1: Set up a python (3.9.18+ recommended) environment, install git and clone the `model-v3` branch from repo

```
git clone -b model-v3 https://github.com/WenxuanLiu3003/4ascend-zero.git
```
> You can also use other method to clone the repo, like the `import from git` option in pyCharm. Just make sure that the imported branch is `model-v3`, instead of `main` or `new-model`.
>
> If you do not know how to install python and git, please ask GPT or Gemini yourself:). Note that to deploy this model you do not need any knowledge on Python.

Step 2: Go to the project directory and setup your own environment and install required packages

```
pip install -r requirements.txt
```
> I strongly suggest using a virtual environment in this step. If you do not know what is virtual environment or how to set up a virtual environment, please ask GPT or Gemini yourself :)

Step 3: Create a new directory to save the checkpoint models under the project path:

```
mkdir checkpoints
```

Then run the game:

```
python -m src.ui.pygame_app
```

If you see a board UI, then your installation is successful.

Step 4: download the model:

go back to [The GitHub release page](https://github.com/WenxuanLiu3003/4ascend-zero/releases), download the most recent `*.pt` model (e.g. `ckpt_trainonly_step75884_1773500722792.pt`). Copy it to your `checkpoints` directory like this:

```
README.md
src
	ai
	core
	replay
	...
checkpoints
	ckpt_trainonly_step75884_1773500722792.pt
```

Step 5: play with AI:

You are all set! You can start play 4ASCEND with AI by

```
python -m src.ui.pygame_app
```

to use the AI, click `run` button.

to edit the plants/black stone/white stone on the board, click the corresponding button. To exit edit mod, click `exit edit`

to back to the last step, click `back`



## How to train my own model?

You can use the following command to perform selfplay:

```
python -m src.train --playOnly
```
After playing enough game and colellecting enough samples, run training
```
python -m src.train --trainOnly
```
It will train the model based on the latest model in `checkpoints/`





---

## Game Summary

4ascend is a turn-based game on a 9×9 board. Players alternate placing stones.  
- When a player forms a chain of **four or more stones**, those stones are **removed** and converted into **“power.”**  
- The opponent can **respond** in an attack–defense phase, attempting to neutralize or counterattack.  
- The **difference in power** determines **HP loss** for one side.  
- **Plants** randomly spawn on empty tiles and grant **+1 power** when included in a chain.  

---


## Acknowledge

Thanks @HashinoMizuha for developing this exciting game!

Thanks @普利姆拉老师 for the source code of the 4ASCEND!
