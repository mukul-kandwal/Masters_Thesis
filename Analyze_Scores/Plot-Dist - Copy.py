#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    if "Score_SMAPE" not in df.columns:
        print("CSV file does not contain the 'Score_SMAPE' column.")
        sys.exit(1)

    # Convert values to float and drop missing/invalid entries
    df["Score_SMAPE"] = pd.to_numeric(df["Score_SMAPE"], errors='coerce')
    scores = df["Score_SMAPE"].dropna().to_numpy()

    if scores.size == 0:
        print("No valid float values found in 'Score_SMAPE'.")
        sys.exit(1)

    # Compute KDE for a smooth density curve
    kde = gaussian_kde(scores)
    x_vals = np.linspace(np.min(scores), np.max(scores), 100)
    y_vals = kde(x_vals)

    # Create a minimalistic figure with a single curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, 
        y=y_vals,
        mode='lines',
        line=dict(color='blue', width=2)
    ))

    # Update layout for a publication-quality figure
    fig.update_layout(
        title={'text': 'Distribution of SMAPE Scores', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Score_SMAPE",
        yaxis_title="Density",
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        font=dict(size=12)
    )

    # Emphasize axis lines with a minimalist style
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, gridcolor='lightgrey')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, gridcolor='lightgrey')

    fig.show()

if __name__ == "__main__":
    main()
