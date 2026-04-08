"""
train.py
--------
Train the Q-learning agent and save the learned Q-table.

Usage
-----
    python train.py
    python train.py --episodes 3000 --max-steps 100 --output results/q_table.npy
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from agent import QLearningAgent, MIN_GREEN
from environment import get_random_state, step


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train RL traffic signal agent")
    p.add_argument("--episodes",  type=int,   default=2500,                  help="Training episodes (default: 2500)")
    p.add_argument("--max-steps", type=int,   default=80,                    help="Max steps per episode (default: 80)")
    p.add_argument("--output",    type=str,   default="results/q_table.npy", help="Where to save the Q-table")
    p.add_argument("--no-plot",   action="store_true",                        help="Skip saving the learning-curve plot")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────
def moving_avg(data, window=50):
    return np.convolve(data, np.ones(window) / window, mode="valid")


def save_learning_curve(rewards, path="results/learning_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg(rewards), color="steelblue", linewidth=1.5, label="50-ep moving avg")
    plt.title("Learning Curve – RL Traffic Signal Control", fontsize=14)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Learning curve saved → {path}")


# ── Training loop ─────────────────────────────────────────────────────────────
def train(episodes: int, max_steps: int, output: str, save_plot: bool = True):
    os.makedirs("results", exist_ok=True)

    agent   = QLearningAgent()
    rewards = []

    for ep in range(episodes):
        state        = get_random_state()
        total_reward = 0.0
        action       = np.random.randint(0, 2)
        green_timer  = 0

        for _ in range(max_steps):
            chosen = agent.select_action(state, action, green_timer)

            # update green timer
            if chosen != action:
                green_timer = 0
            else:
                green_timer += 1

            next_state, reward = step(state, chosen)
            agent.update(state, chosen, reward, next_state)

            state        = next_state
            action       = chosen
            total_reward += reward

        rewards.append(total_reward)
        agent.decay_epsilon()

        if (ep + 1) % 250 == 0:
            print(f"Episode {ep+1:>5}/{episodes}  |  "
                  f"ε={agent.epsilon:.4f}  |  "
                  f"Avg reward (last 50): {np.mean(rewards[-50:]):.1f}")

    agent.save(output)

    if save_plot:
        save_learning_curve(rewards)

    return agent, rewards


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train(
        episodes  = args.episodes,
        max_steps = args.max_steps,
        output    = args.output,
        save_plot = not args.no_plot,
    )
