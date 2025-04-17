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
Y_LABEL = "Regression Models"  # General y-axis label
TITLE = sys.argv[2]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
FIG_SIZE = (10, 6)
OUTPUT_NAME = "plot.png"

# Specific y-axis labels mapping
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

def plot_data(df):
    """Create custom raincloud-style plot"""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Sort and reset index
    df = df.sort_values('Score_SMAPE', ascending=True).reset_index(drop=True)
    
    # Plot elements for each method
    for i, row in df.iterrows():
        residuals = row['Score_SMAPE_Residuals']
        score = row['Score_SMAPE']
        
        # Vertical position (inverted for top-to-bottom ordering)
        y_pos = len(df) - i - 1
        
        # Individual points
        y_jitter = np.random.normal(y_pos, 0.1, len(residuals))
        ax.scatter(
            residuals, y_jitter,
            alpha=0.4, color=COLORS[i], s=20,
            edgecolor='none'
        )
        
        # Distribution visualization
        ax.violinplot(
            residuals, positions=[y_pos], 
            vert=False, widths=0.7,
            showmeans=False, showextrema=False
        )
        
        # SMAPE score line
        ax.vlines(
            score, 
            y_pos - 0.3,
            y_pos + 0.3,
            color='black', 
            linestyle='--',
            linewidth=1.5
        )
    
    # Configure axes
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([LABEL_MAP[m] for m in reversed(df['Differencing'])])
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)  # Set general y-axis label here
    ax.set_title(TITLE)
    
    # Invert y-axis for correct ordering
    ax.invert_yaxis()
    
    # Style adjustments
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2)
    
    # Save output
    output_path = os.path.join(os.path.dirname(sys.argv[1]), OUTPUT_NAME)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {output_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input.csv> <graph title>")
        sys.exit(1)
    
    try:
        # Read and process data
        df = pd.read_csv(sys.argv[1])
        df['Score_SMAPE_Residuals'] = df['Score_SMAPE_Residuals'].apply(
            lambda x: [float(n) for n in ast.literal_eval(x)]
        )
        plot_data(df)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()