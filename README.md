# 4ascend-Zero: AlphaZero-Style AI for the 4ascend Game

## Overview
**4ascend-Zero** is an AlphaZero-style reinforcement learning system built for the **4ascend board game**, a 9×9 strategy game featuring dynamic attack–defense mechanics, plant-based resource tiles, and HP-based victory conditions. (see [4ASCEND | フリーゲーム投稿サイト unityroom](https://unityroom.com/games/4ascend))

The project combines:
- **Monte Carlo Tree Search (MCTS)**
- **Deep Residual Policy–Value Networks**
- **Self-play Reinforcement Learning**

The goal is to train an agent capable of mastering both **positional tactics** and **strategic resource control** under a stochastic environment.

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

