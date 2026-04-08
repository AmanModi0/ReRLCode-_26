"""
Q-Learning Agent for Traffic Signal Control
--------------------------------------------
Tabular Q-learning with epsilon-greedy exploration
and a minimum green-time constraint.
"""

import numpy as np
import random

from environment import BINS, ACTIONS, discretize

# ── Hyperparameters ──────────────────────────────────────────────────────────
ALPHA         = 0.1      # learning rate
GAMMA         = 0.95     # discount factor
EPSILON_START = 1.0      # initial exploration rate
EPSILON_MIN   = 0.05     # floor for exploration
EPSILON_DECAY = 0.993    # multiplicative decay per episode
MIN_GREEN     = 5        # minimum steps before a phase switch is allowed


class QLearningAgent:
    """Tabular Q-learning agent for a 4-way intersection."""

    def __init__(self):
        # Q-table: state = (bin_n, bin_s, bin_e, bin_w), action ∈ {0, 1}
        self.Q       = np.zeros((BINS, BINS, BINS, BINS, ACTIONS))
        self.epsilon = EPSILON_START

    # ── Internal helpers ─────────────────────────────────────────────────
    @staticmethod
    def _disc(state: tuple) -> tuple:
        return tuple(discretize(x) for x in state)

    # ── Action selection ─────────────────────────────────────────────────
    def select_action(self, state: tuple, current_action: int, green_timer: int) -> int:
        """
        Choose the next action respecting the minimum-green-time rule.

        Parameters
        ----------
        state          : raw (n, s, e, w) queue lengths
        current_action : the phase currently active
        green_timer    : steps elapsed under the current phase

        Returns
        -------
        chosen action (int)
        """
        if green_timer < MIN_GREEN:
            return current_action          # enforce minimum green

        ds = self._disc(state)
        if random.random() < self.epsilon:
            return random.randint(0, ACTIONS - 1)
        return int(np.argmax(self.Q[ds]))

    # ── Q update ─────────────────────────────────────────────────────────
    def update(self, state: tuple, action: int, reward: float, next_state: tuple):
        ds      = self._disc(state)
        ds_next = self._disc(next_state)
        td_target = reward + GAMMA * np.max(self.Q[ds_next])
        self.Q[ds][action] += ALPHA * (td_target - self.Q[ds][action])

    # ── Epsilon decay ────────────────────────────────────────────────────
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    # ── Persistence ──────────────────────────────────────────────────────
    def save(self, path: str = "results/q_table.npy"):
        np.save(path, self.Q)
        print(f"Q-table saved → {path}")

    def load(self, path: str = "results/q_table.npy"):
        self.Q = np.load(path)
        self.epsilon = EPSILON_MIN      # evaluation mode: minimal exploration
        print(f"Q-table loaded ← {path}")
