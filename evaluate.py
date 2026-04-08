"""
evaluate.py
-----------
Compare the trained RL agent against a naive fixed-timer controller.

Usage
-----
    python evaluate.py
    python evaluate.py --q-table results/q_table.npy --trials 300
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
    p = argparse.ArgumentParser(description="Evaluate RL vs Fixed-timer controller")
    p.add_argument("--q-table",   default="results/q_table.npy", help="Path to saved Q-table")
    p.add_argument("--trials",    type=int, default=300,         help="Evaluation episodes (default: 300)")
    p.add_argument("--max-steps", type=int, default=80,          help="Steps per episode (default: 80)")
    p.add_argument("--no-plot",   action="store_true",           help="Skip saving bar chart")
    return p.parse_args()


# ── Baselines ─────────────────────────────────────────────────────────────────
def run_fixed(max_steps: int = 80, switch_every: int = 10) -> float:
    """Fixed-cycle controller: switches phase every `switch_every` steps."""
    state      = get_random_state()
    total_wait = 0.0
    action     = 0
    timer      = 0

    for _ in range(max_steps):
        if timer >= switch_every:
            action = 1 - action
            timer  = 0
        _, reward  = step(state, action)
        state, _   = step(state, action)   # advance state
        total_wait += -reward
        timer      += 1

    return total_wait


def run_rl(agent: QLearningAgent, max_steps: int = 80) -> float:
    """RL agent controller (greedy policy, minimum green enforced)."""
    state      = get_random_state()
    total_wait = 0.0
    action     = 0
    timer      = 0

    for _ in range(max_steps):
        chosen = agent.select_action(state, action, timer)
        if chosen != action:
            timer = 0
        else:
            timer += 1

        next_state, reward = step(state, chosen)
        total_wait += -reward
        state       = next_state
        action      = chosen

    return total_wait


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(q_table_path: str, trials: int, max_steps: int, save_plot: bool = True):
    os.makedirs("results", exist_ok=True)

    # Load agent
    agent = QLearningAgent()
    agent.load(q_table_path)

    # Run trials
    fixed_results = [run_fixed(max_steps) for _ in range(trials)]
    rl_results    = [run_rl(agent, max_steps) for _ in range(trials)]

    fixed_mean, fixed_std = np.mean(fixed_results), np.std(fixed_results)
    rl_mean,    rl_std    = np.mean(rl_results),    np.std(rl_results)
    improvement           = (fixed_mean - rl_mean) / fixed_mean * 100

    print("\n" + "=" * 40)
    print("       EVALUATION RESULTS")
    print("=" * 40)
    print(f"  Fixed timer : {fixed_mean:.2f}  (±{fixed_std:.2f})")
    print(f"  RL agent    : {rl_mean:.2f}  (±{rl_std:.2f})")
    print(f"  Improvement : {improvement:.2f}%")
    print("=" * 40)

    if save_plot:
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(
            ["Fixed Timer", "RL Agent"],
            [fixed_mean, rl_mean],
            yerr=[fixed_std, rl_std],
            color=["#e07b54", "#4c8fbd"],
            capsize=8,
            width=0.5,
        )
        ax.set_title("Traffic Signal Control – Performance Comparison", fontsize=13)
        ax.set_ylabel("Total Waiting Cost (lower is better)")
        ax.set_ylim(0, max(fixed_mean, rl_mean) * 1.3)

        for bar, val in zip(bars, [fixed_mean, rl_mean]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                f"{val:.1f}",
                ha="center", va="bottom", fontweight="bold",
            )

        ax.annotate(
            f"RL improves by {improvement:.1f}%",
            xy=(0.5, 0.92), xycoords="axes fraction",
            ha="center", fontsize=11, color="green",
        )
        plt.tight_layout()
        out = "results/comparison.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Bar chart saved → {out}")

    return fixed_mean, rl_mean, improvement


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    evaluate(
        q_table_path = args.q_table,
        trials       = args.trials,
        max_steps    = args.max_steps,
        save_plot    = not args.no_plot,
    )
