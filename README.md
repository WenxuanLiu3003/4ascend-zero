# 4ascend-Zero: AlphaZero-Style AI for the 4ascend Game

## Overview
**4ascend-Zero** is an AlphaZero-style reinforcement learning system built for the **4ascend board game**, a 9×9 strategy game featuring dynamic attack–defense mechanics, plant-based resource tiles, and HP-based victory conditions. (see [4ASCEND | フリーゲーム投稿サイト unityroom](https://unityroom.com/games/4ascend))

The project combines:
- **Monte Carlo Tree Search (MCTS)**
- **Deep Residual Policy–Value Networks**
- **Self-play Reinforcement Learning**

The goal is to train an agent capable of mastering both **positional tactics** and **strategic resource control** under a stochastic environment.

---
## How to use the code for training?
clone the `new-model` branch from repo
```
git clone -b new-model https://github.com/WenxuanLiu3003/4ascend-zero.git
```
Go to the project directory and setup your own environment and install required packages
```
pip install -r requirements.txt
```
Create a new directory to save the checkpoint models:
```
mkdir checkpoints
```

Then you can use the following command to train:
```
python -m src.train
```
We also provides the following command parameters to adjust the training parameters
```
--epoch=1: the training epoch
--sim=1000: number of simulations for each step in selfplaying
--game=500: number of selfplaying games per epoch
--batch=256: batch number for training 
```
For example, if you want to set game=1000 and run 2 epochs, you should run the following command
```
python -m src.train --epoch 2 --game 500
```

You can watch two AI players to play the game if there are existing model file (*.pt) in `/checkpoints`.
```
python -m src.replay.run_gui_selfplay
```




---

## Game Summary
4ascend is a turn-based game on a 9×9 board. Players alternate placing stones.  
- When a player forms a chain of **four or more stones**, those stones are **removed** and converted into **“power.”**  
- The opponent can **respond** in an attack–defense phase, attempting to neutralize or counterattack.  
- The **difference in power** determines **HP loss** for one side.  
- **Plants** randomly spawn on empty tiles and grant **+1 power** when included in a chain.  

---

## Architecture

### 1. Neural Network
The AI uses a **ResNet-style policy–value network**:
- Input: current board state (stones, plants, HP, phase indicators, etc.)
- Output:  
  - **Policy head** → probability distribution over 81 possible moves  
  - **Value head** → scalar in [-1, 1] representing the expected game outcome

### 2. Monte Carlo Tree Search (MCTS)
At each move, the agent performs **PUCT-based search** using the network’s policy and value predictions to guide simulations.  
Key features:
- **Dirichlet noise** at root for exploration  
- **Move legality masking** (dynamic during attack–defense phases)  
- **Plant regeneration modeling** as stochastic environment transitions

### 3. Self-Play Training Loop
The agent continually improves via self-play:
1. **Self-play** → generate trajectories with MCTS-enhanced actions  
2. **Replay buffer** → store (state, MCTS policy, outcome) tuples  
3. **Network training** → minimize combined policy + value loss  
4. **Evaluation** → play against previous versions to ensure progress  

---

## Project Structure
```
4ascend-zero/
│
├── src/
│   ├── engine/             # Game rules, state transitions, legality checks
│   ├── mcts/               # Monte Carlo Tree Search implementation
│   ├── selfplay/           # Self-play worker for data generation
│   ├── models/             # Policy–value network definitions
│   ├── train.py            # Main training loop
│   ├── evaluate.py         # Elo rating & performance testing
│   └── visualize.py        # Optional GUI or web-based viewer
│
├── data/
│   ├── replay_buffer/      # Self-play game records
│   └── checkpoints/        # Saved neural network weights
│
├── 4ascend_rules_en.md     # Official game rules (English)
└── README.md               # You are here
```

