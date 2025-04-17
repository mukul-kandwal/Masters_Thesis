import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

# ======================
# CONFIGURATION SECTION
# ======================
X_LABEL = "SMAPE Residuals"
Y_LABEL = "Regression Models"
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
FIG_SIZE = (10, 6)
OUTPUT_NAME = "plot.png"

LABEL_MAP = {
    "Default": "None",
    "ABS, 0": "None",
    "ABS, 1": "Abs, Order 1",
    "ABS, 2": "Abs, Order 2",
    "Rel, 1": "Rel, Order 1",
    "Rel, 2": "Rel, Order 2",
    "Baseline": "Naive Control",
    "XGBoost": "XGBoost",
    "RF": "Random Forest",
    "NN": "Neural Network",
    "KNN": "K-Nearest Neighbours",
}
# ======================

def plot_data(df, title, bin_width):
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # sort so first ends up at top
    df = df.sort_values('Score_SMAPE', ascending=True).reset_index(drop=True)

    # common bins
    all_res = [r for row in df['Score_SMAPE_Residuals'] for r in row]
    bins = np.arange(min(all_res), max(all_res) + bin_width, bin_width)

    for i, row in df.iterrows():
        residuals = row['Score_SMAPE_Residuals']
        score = row['Score_SMAPE']          # mean
        median_val = np.median(residuals)   # <— compute median
        y_pos = len(df) - i - 1

        counts, _ = np.histogram(residuals, bins=bins)
        if counts.max() > 0:
            counts = counts / counts.max() * 0.8

        # ─── draw bars that extend UP visually by using negative heights ─────────
        ax.bar(
            bins[:-1],
            -counts,           # negative so bar goes “up” on screen
            width=bin_width,
            bottom=y_pos,      # baseline
            align='edge',
            color=COLORS[i % len(COLORS)],
            alpha=0.6,
            edgecolor='none'
        )

        # ─── draw the dotted line for the MEAN ───────────────────────────────────
        ax.vlines(
            score,
            y_pos,             # start at baseline
            y_pos - 0.8,       # go “up” by 0.8 in display coords
            linestyle='--',
            linewidth=1.0,
            color="black"
        )

        # ─── draw the solid line for the MEDIAN ─────────────────────────────────
        ax.vlines(
            median_val,
            y_pos,
            y_pos - 0.8,
            linestyle='-',
            linewidth=1.0,
            color="black"
        )

    # …rest of your styling/config (unchanged)…
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([LABEL_MAP[m] for m in reversed(df['Differencing'])])
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='both', alpha=0.2)

    plt.savefig(os.path.join(os.path.dirname(sys.argv[1]), OUTPUT_NAME),
                bbox_inches='tight', dpi=300)
    print(f"Plot saved")

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <input.csv> <graph title> <bin_width>")
        sys.exit(1)

    input_csv, title, bin_arg = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        bin_width = float(bin_arg)
    except ValueError:
        print("Error: bin_width must be a number (e.g. 0.1)")
        sys.exit(1)

    try:
        df = pd.read_csv(input_csv)
        df['Score_SMAPE_Residuals'] = df['Score_SMAPE_Residuals'].apply(
            lambda x: [float(n) for n in ast.literal_eval(x)]
        )
        plot_data(df, title, bin_width)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()