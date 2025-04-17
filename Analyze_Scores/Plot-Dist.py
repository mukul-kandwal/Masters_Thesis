import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Check command-line arguments: expect input CSV file and number of bins.
if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <input_csv_file> <number_of_bins>")
    sys.exit(1)

input_file = sys.argv[1]

try:
    number_of_bins = int(sys.argv[2])
except ValueError:
    print("Error: Number of bins must be an integer.")
    sys.exit(1)

try:
    # Read CSV file
    df = pd.read_csv(input_file)
    
    # Extract Score_SMAPE column and remove missing values
    scores = df['Score_SMAPE'].dropna().tolist()
    
except FileNotFoundError:
    print(f"Error: File '{input_file}' not found.")
    sys.exit(1)
except KeyError:
    print("Error: CSV file does not contain 'Score_SMAPE' column.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {str(e)}")
    sys.exit(1)

# Check if we have valid data
if not scores:
    print("Error: No valid data found in 'Score_SMAPE' column.")
    sys.exit(1)

# Compute histogram with counts (density normalization turned off)
hist, bin_edges = np.histogram(scores, bins=number_of_bins, density=False)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]

# Create the histogram as a bar chart.
fig = go.Figure(data=[
    go.Bar(
        x=bin_centers,
        y=hist,
        width=bin_width * 0.9,  # Slightly narrow the bars for visual separation
        marker=dict(
            color='rgba(75, 0, 130, 0.2)',  # lighter fill color
            line=dict(color='#4B0082', width=2)  # outline with the original line color
        ),
        hoverinfo='x+y'
    )
])

# Customize layout for a clean, thesis-ready aesthetic.
fig.update_layout(
    title={'text': '(SMAPE) Score Distribution', 'x': 0.5, 'xanchor': 'center'},
    xaxis_title='Score',
    yaxis_title='Count',
    template='simple_white',
    hovermode='x unified',
    showlegend=False,
    font=dict(
        family="Arial",
        size=14,
        color="black"
    ),
    margin=dict(l=60, r=60, t=60, b=60)
)

# Update x-axis: ensure a thick black line and clear zeroline.
fig.update_xaxes(
    showline=True,
    linecolor='black',
    linewidth=2,
    zeroline=True,
    zerolinecolor='black',
    zerolinewidth=2
)

# Update y-axis: match the x-axis line style.
fig.update_yaxes(
    showline=True,
    linecolor='black',
    linewidth=2,
    zeroline=True,
    zerolinecolor='black',
    zerolinewidth=2
)

# Removed the x-axis start marker annotation.

# Show the plot.
fig.show()
