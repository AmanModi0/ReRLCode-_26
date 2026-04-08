# RL Traffic Signal Control

A reinforcement learning agent that learns to control a 4-way traffic intersection using **tabular Q-learning**. The agent significantly outperforms a naive fixed-timer controller by dynamically responding to real-time queue imbalances.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Algorithm Details](#algorithm-details)
- [Future Work](#future-work)

---

## Overview

Traditional traffic lights switch on fixed cycles regardless of actual traffic conditions. This project trains a Q-learning agent to **adaptively control signal phases** at a single 4-way intersection, minimising total vehicle waiting time.

| Controller   | Strategy                         |
|--------------|----------------------------------|
| Fixed Timer  | Switches phase every 10 steps    |
| **RL Agent** | Learns optimal switching policy  |

---

## How It Works

### Environment

The intersection has **4 queues** (North, South, East, West). At each time step:

1. The agent picks an action: **0 = N/S green** or **1 = E/W green**
2. Up to 10 cars per active direction clear the intersection
3. New cars arrive via a **Poisson process** (λ=2 per direction)
4. A **pressure-based reward** is computed

### Reward Function

```
reward = (cars_cleared × 3)
       - (total_queue_length × 1.5)
       - (|NS_queue − EW_queue| × 2)
```

This incentivises throughput, penalises congestion, and reduces directional imbalance.

### Minimum Green Time

To reflect real-world constraints, the agent cannot switch phase until at least **5 consecutive steps** have elapsed under the current phase.

---

## Project Structure

```
smart-traffic-control-rl/
├── environment.py        # Intersection simulation & reward function
├── agent.py              # Q-learning agent (select, update, save/load)
├── train.py              # Training entry point
├── evaluate.py           # Evaluation & Fixed vs RL comparison
├── inspect_qtable.py     # Visualise & export the saved Q-table
├── test_environment.py   # Unit tests (pytest)
├── results/              # Saved Q-tables & plots (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/smart-traffic-control-rl.git
cd smart-traffic-control-rl

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip3 install -r requirements.txt
```

---

## Usage

### Train

```bash
python3 train.py
```

Optional flags:

| Flag            | Default                  | Description                    |
|-----------------|--------------------------|--------------------------------|
| `--episodes`    | `2500`                   | Number of training episodes    |
| `--max-steps`   | `80`                     | Steps per episode              |
| `--output`      | `results/q_table.npy`    | Path to save the Q-table       |
| `--no-plot`     | *(off)*                  | Skip saving learning curve     |

```bash
# Example: longer training run
python3 train.py --episodes 5000 --max-steps 120
```

### Evaluate

```bash
python3 evaluate.py
```

Optional flags:

| Flag            | Default                  | Description                         |
|-----------------|--------------------------|-------------------------------------|
| `--q-table`     | `results/q_table.npy`    | Path to a saved Q-table             |
| `--trials`      | `300`                    | Number of evaluation episodes       |
| `--max-steps`   | `80`                     | Steps per episode                   |
| `--no-plot`     | *(off)*                  | Skip saving bar chart               |

### Inspect Q-Table

The Q-table is saved as a `.npy` binary file. To view it in a human-readable format:

```bash
python3 inspect_qtable.py
```

This generates 3 files in `results/`:

| File | Description |
|------|-------------|
| `q_table.csv` | All Q-values as a CSV — open in Excel or Google Sheets |
| `q_table_actions.png` | Bar chart of preferred actions across all states |
| `q_table_heatmap.png` | Heatmap of max Q-values across queue states |

It also prints a summary in the terminal:
```
========== Q-TABLE SUMMARY ==========
  Shape        : (8, 8, 8, 8, 2)
  Total states : 4096
  Max Q-value  : 4.4330
  Min Q-value  : -301.8664
  Mean Q-value : -11.9761
======================================
```

### Run Tests

```bash
pytest test_environment.py -v
```

---

## Results

After 2500 training episodes the RL agent consistently outperforms the fixed-timer baseline:

| Metric             |     Fixed Timer    |      RL Agent     |
|--------------------|--------------------|-------------------|
| Avg waiting cost   |  6548.35 (±458.62) | 3493.34 (±315.20) |
| Improvement        |          —         |     **46.65%**    |


Generated plots are saved to `results/`:

- `learning_curve.png` — reward vs episode (50-episode moving average)
- `comparison.png`     — bar chart: Fixed vs RL waiting cost
- `q_table_actions.png` — preferred action distribution
- `q_table_heatmap.png` — Q-value heatmap across states

---

## Algorithm Details

| Component             | Value / Method                              |
|-----------------------|---------------------------------------------|
| Algorithm             | Tabular Q-learning                          |
| State space           | 4 queues × 8 bins each = 8⁴ = 4,096 states |
| Action space          | 2 (N/S green, E/W green)                   |
| Learning rate α       | 0.1                                         |
| Discount factor γ     | 0.95                                        |
| Exploration           | ε-greedy, ε: 1.0 → 0.05 (decay 0.993)     |
| Min green time        | 5 steps                                     |
| Arrivals              | Poisson (λ=2 per direction)                 |
| Max queue depth       | 30 cars per direction                       |

