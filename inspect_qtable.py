"""
inspect_qtable.py
-----------------
Load and visualise the saved Q-table in human-readable formats.
Exports:
  - results/q_table.csv        (flat CSV, openable in Excel / Sheets)
  - results/q_table_actions.png (histogram of preferred actions)
  - results/q_table_heatmap.png (heatmap of max Q-values)

Usage
-----
    python3 inspect_qtable.py
    python3 inspect_qtable.py --q-table results/q_table.npy
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Inspect saved Q-table")
    p.add_argument("--q-table", default="results/q_table.npy", help="Path to Q-table .npy file")
    return p.parse_args()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    # Load
    q = np.load(args.q_table)
    print("\n========== Q-TABLE SUMMARY ==========")
    print(f"  Shape        : {q.shape}  (bins_n, bins_s, bins_e, bins_w, actions)")
    print(f"  Total states : {q.shape[0]**4}")
    print(f"  Max Q-value  : {q.max():.4f}")
    print(f"  Min Q-value  : {q.min():.4f}")
    print(f"  Mean Q-value : {q.mean():.4f}")
    print("======================================\n")

    # ── 1. Export to CSV ──────────────────────────────────────────────────
    q_flat = q.reshape(-1, 2)
    csv_path = "results/q_table.csv"
    np.savetxt(csv_path, q_flat, delimiter=",",
               header="q_action_NS_green,q_action_EW_green", comments="")
    print(f"CSV saved → {csv_path}  (open in Excel or Google Sheets)")

    # ── 2. Preferred action histogram ─────────────────────────────────────
    best_actions = np.argmax(q, axis=-1).flatten()
    ns_pct = (best_actions == 0).mean() * 100
    ew_pct = (best_actions == 1).mean() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["N/S Green", "E/W Green"], [ns_pct, ew_pct],
           color=["#4c8fbd", "#e07b54"], edgecolor="black", width=0.5)
    ax.set_ylabel("% of states where action is preferred")
    ax.set_title("Preferred Action Distribution Across All States")
    for i, v in enumerate([ns_pct, ew_pct]):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylim(0, 110)
    plt.tight_layout()
    hist_path = "results/q_table_actions.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Action histogram saved → {hist_path}")

    # ── 3. Heatmap of max Q-values (collapsed to 2D) ──────────────────────
    # Average over all N and W bins → (bins_s, bins_e) slice
    q_2d = q.max(axis=-1).mean(axis=(0, 3))   # shape: (bins_s, bins_e)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(q_2d, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Max Q-value")
    ax.set_title("Max Q-value Heatmap\n(averaged over N & W queues)")
    ax.set_xlabel("East queue bin")
    ax.set_ylabel("South queue bin")
    plt.tight_layout()
    heatmap_path = "results/q_table_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Heatmap saved → {heatmap_path}")

    print("\nDone! Check the results/ folder.")

if __name__ == "__main__":
    main()
