# 4ascend-Zero: Alphago-Zero-Style AI for the 4ascend Game

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
