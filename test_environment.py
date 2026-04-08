"""
tests/test_environment.py
--------------------------
Unit tests for the traffic signal environment.
"""

import pytest
import numpy as np

from src.environment import discretize, get_random_state, step, MAX_CARS, BINS


# ── discretize ────────────────────────────────────────────────────────────────
class TestDiscretize:
    def test_zero(self):
        assert discretize(0) == 0

    def test_max_bin(self):
        assert discretize(MAX_CARS) == BINS - 1

    def test_mid(self):
        assert discretize(16) == 4

    def test_clamps_to_bins_minus_one(self):
        assert discretize(999) == BINS - 1


# ── get_random_state ──────────────────────────────────────────────────────────
class TestGetRandomState:
    def test_length(self):
        s = get_random_state()
        assert len(s) == 4

    def test_bounds(self):
        for _ in range(50):
            s = get_random_state()
            assert all(0 <= v <= MAX_CARS for v in s)


# ── step ──────────────────────────────────────────────────────────────────────
class TestStep:
    def test_returns_tuple_and_float(self):
        ns, r = step((5, 5, 5, 5), 0)
        assert isinstance(ns, tuple)
        assert len(ns) == 4
        assert isinstance(r, float)

    def test_queues_capped_at_max(self):
        for _ in range(20):
            ns, _ = step((MAX_CARS, MAX_CARS, MAX_CARS, MAX_CARS), 0)
            assert all(v <= MAX_CARS for v in ns)

    def test_queues_non_negative(self):
        for _ in range(20):
            ns, _ = step((0, 0, 0, 0), 1)
            assert all(v >= 0 for v in ns)

    def test_action_0_clears_ns(self):
        """With action 0, N/S queues should decrease (before arrivals)."""
        np.random.seed(0)
        # Force arrivals to 0 by patching (pragmatic: just run many trials)
        decreases = 0
        for _ in range(30):
            n0, s0 = 20, 20
            (n1, s1, _, _), _ = step((n0, s0, 5, 5), 0)
            if n1 < n0 or s1 < s0:
                decreases += 1
        assert decreases > 0

    def test_reward_penalises_high_queues(self):
        _, r_low  = step((2, 2, 2, 2), 0)
        _, r_high = step((28, 28, 28, 28), 0)
        assert r_high < r_low


# ── agent smoke test ──────────────────────────────────────────────────────────
class TestAgent:
    def test_agent_trains_one_episode(self):
        from src.agent import QLearningAgent, MIN_GREEN
        import random

        agent  = QLearningAgent()
        state  = get_random_state()
        action = random.randint(0, 1)
        timer  = 0

        for _ in range(20):
            chosen = agent.select_action(state, action, timer)
            assert chosen in (0, 1)
            ns, reward = step(state, chosen)
            agent.update(state, chosen, reward, ns)
            if chosen != action:
                timer = 0
            else:
                timer += 1
            state  = ns
            action = chosen

        agent.decay_epsilon()
        assert agent.epsilon < 1.0
