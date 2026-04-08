"""
Traffic Signal Environment
--------------------------
Simulates a 4-way intersection with N/S/E/W traffic queues.
"""

import numpy as np
import random

MAX_CARS = 30
ACTIONS = 2      # 0 = N/S green, 1 = E/W green
BINS = 8


def discretize(x: int) -> int:
    """Discretize a queue length into a bin index (0 – BINS-1)."""
    return min(x // 4, BINS - 1)


def get_random_state() -> tuple:
    """Return a random initial (n, s, e, w) queue-length tuple."""
    return tuple(random.randint(0, MAX_CARS) for _ in range(4))


def step(state: tuple, action: int) -> tuple:
    """
    Advance the simulation by one time step.

    Parameters
    ----------
    state  : (n, s, e, w) current queue lengths
    action : 0 → N/S green, 1 → E/W green

    Returns
    -------
    next_state : updated (n, s, e, w)
    reward     : scalar reward signal
    """
    n, s, e, w = state

    # ── Cars clearing the intersection ──────────────────────────────────
    if action == 0:
        move_n, move_s = min(n, 10), min(s, 10)
        n -= move_n
        s -= move_s
        passed = move_n + move_s
    else:
        move_e, move_w = min(e, 10), min(w, 10)
        e -= move_e
        w -= move_w
        passed = move_e + move_w

    # ── Stochastic arrivals (Poisson λ=2 per direction) ─────────────────
    n += np.random.poisson(2)
    s += np.random.poisson(2)
    e += np.random.poisson(2)
    w += np.random.poisson(2)

    n, s, e, w = [min(x, MAX_CARS) for x in (n, s, e, w)]

    next_state = (n, s, e, w)

    # ── Pressure-based reward ────────────────────────────────────────────
    imbalance = abs((n + s) - (e + w))
    reward = (
        passed * 3                  # reward cars cleared
        - (n + s + e + w) * 1.5    # penalise total queue depth
        - imbalance * 2             # penalise directional imbalance
    )

    return next_state, reward
