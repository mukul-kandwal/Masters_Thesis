#!/usr/bin/env python3
import os
import sys
import ast
import pandas as pd
import plotly.graph_objects as go

def calculate_predictions(target, r):
    """
    Given a SMAPE residual 'r' and a target value 'target',
    return the two possible predictions that would yield that SMAPE residual.
    
    SMAPE is defined as:
        SMAPE = 200 * |N - p| / (|N| + |p|)
    
    For the two cases:
        p_below = target * (200 - r) / (200 + r)
        p_above = target * (200 + r) / (200 - r)
    
    Note: if r == 0, both formulas return target.
    """
    if r == 200:
        p_below = target * (200 - r) / (200 + r)  # yields 0
        p_above = float('inf')
    else:
        p_below = target * (200 - r) / (200 + r)
        p_above = target * (200 + r) / (200 - r)
    return p_below, p_above

def process_csv(file_path, target, rows_to_plot, print_flag):
    """Reads a CSV file, filters rows based on Score_SMAPE, and creates a Plotly figure."""
    print(f"Processing {file_path}")
    df = pd.read_csv(file_path)
    
    # Sort by the "Score_SMAPE" column (if present) and select the specified number of rows.
    if "Score_SMAPE" in df.columns:
        try:
            df["Score_SMAPE"] = pd.to_numeric(df["Score_SMAPE"], errors="coerce")
            df = df.sort_values(by="Score_SMAPE", ascending=True).head(rows_to_plot)
        except Exception as e:
            print(f"Error processing 'Score_SMAPE' column in {file_path}: {e}", file=sys.stderr)
    else:
        print(f"Warning: 'Score_SMAPE' column not found in {file_path}. Proceeding with all rows.", file=sys.stderr)
    
    scatter_points = []  # To hold red dot points
    violin_traces = []   # One violin trace per row
    
    # Process each selected row
    for i, row in df.iterrows():
        residuals_str = row.get("Score_SMAPE_Residuals", "[]")
        try:
            residuals = ast.literal_eval(residuals_str)
        except Exception as e:
            print(f"Error parsing SMAPE residuals in row {i+1}: {residuals_str}", file=sys.stderr)
            continue
        
        predictions = []
        for r in residuals:
            try:
                r_val = float(r)
            except Exception as e:
                continue
            p_lower, p_upper = calculate_predictions(target, r_val)
            predictions.extend([p_lower, p_upper])
        
        # Use the row number (as string) for categorical grouping on the x-axis.
        row_label = str(i+1)
        for pred in predictions:
            scatter_points.append(dict(x=row_label, y=pred))
        
        # Create a violin plot trace for this row (blue)
        violin_trace = go.Violin(
            y=predictions,
            x=[row_label] * len(predictions),
            name=row_label,
            marker=dict(color='blue'),
            showlegend=False,
            points=False  # Disable default individual points within the violin
        )
        violin_traces.append(violin_trace)
    
    # Create a scatter trace for all computed prediction points (red dots)
    scatter_trace = go.Scatter(
        x=[pt["x"] for pt in scatter_points],
        y=[pt["y"] for pt in scatter_points],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Predicted Points'
    )
    
    # Build the figure with both scatter and violin traces.
    fig = go.Figure()
    fig.add_trace(scatter_trace)
    for violin in violin_traces:
        fig.add_trace(violin)
    
    # Add a horizontal dashed green reference line at y = target.
    # Use paper coordinates for x so it spans the entire width.
    fig.add_shape(
        type="line",
        xref="paper",
        yref="y",
        x0=0,
        x1=1,
        y0=target,
        y1=target,
        line=dict(color="green", dash="dash")
    )
    
    # Configure the layout: set the x-axis as categorical so that labels are evenly spaced.
    fig.update_layout(
        title=f"SMAPE-Derived Predictions for {os.path.basename(file_path)}",
        xaxis_title="Row Number",
        yaxis_title="Predicted Value",
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=[str(i) for i in range(1, len(df)+1)]
        ),
        margin=dict(b=100)  # Extra bottom margin to ensure labels are visible
    )
    
    # Save the figure as an HTML file with the CSV's basename.
    html_filename = os.path.splitext(os.path.basename(file_path))[0] + ".html"
    fig.write_html(html_filename)
    print(f"Saved plot to {html_filename}")
    
    # Optionally display the plot if print_flag is True.
    if print_flag:
        fig.show()

def main():
    if len(sys.argv) < 5:
        print("Usage: python plot_smapes.py <path_to_csv_or_folder> <target> <rows_to_plot> <print_flag(true/false)>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    try:
        target = float(sys.argv[2])
    except ValueError:
        print("Target must be a number.", file=sys.stderr)
        sys.exit(1)
    
    try:
        rows_to_plot = int(sys.argv[3])
    except ValueError:
        print("Rows to plot must be an integer.", file=sys.stderr)
        sys.exit(1)
    
    print_flag_str = sys.argv[4].lower()
    print_flag = print_flag_str == "true"
    
    if os.path.isdir(input_path):
        # Process all CSV files in the folder ending with 'SeasonGrouped.csv'
        for f in os.listdir(input_path):
            if f.endswith("SeasonGrouped.csv"):
                full_path = os.path.join(input_path, f)
                process_csv(full_path, target, rows_to_plot, print_flag)
    elif os.path.isfile(input_path):
        process_csv(input_path, target, rows_to_plot, print_flag)
    else:
        print("The provided path is neither a file nor a directory.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
