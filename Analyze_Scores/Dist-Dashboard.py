import sys
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
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

# Read CSV file and get the data.
try:
    df = pd.read_csv(input_file)
    # Ensure Score_SMAPE is numeric and drop missing values for histogram computation.
    df['Score_SMAPE'] = pd.to_numeric(df['Score_SMAPE'], errors='coerce')
    score_data = df['Score_SMAPE'].dropna().values
except FileNotFoundError:
    print(f"Error: File '{input_file}' not found.")
    sys.exit(1)
except KeyError:
    print("Error: CSV file does not contain 'Score_SMAPE' column.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {str(e)}")
    sys.exit(1)

# Compute common bin edges for the histogram based on all Score_SMAPE values.
_, bin_edges = np.histogram(score_data, bins=number_of_bins, density=False)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]

# Define a standardized color palette (including the current color and additional contrasting colors).
color_palette = ['#4B0082', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']

# Create a list of candidate metadata columns (those that start with "metadata_").
candidate_columns = [col for col in df.columns if col.startswith("metadata_")]

# Define a helper function to clean column names for display.
def clean_column_name(col):
    col = col.replace("metadata_", "")  # Remove prefix.
    col = col.replace("_", " ")           # Replace underscores with spaces.
    return col.title()                    # Title-case the result.

# Prepare dropdown options.
dropdown_options = [{'label': clean_column_name(col), 'value': col} for col in candidate_columns]

# Create Dash app layout.
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("(SMAPE) Score Distribution", style={'textAlign': 'center', 'fontFamily': 'Arial', 'color': 'black'}),
    html.Div([
        html.Label("Select metadata column:", style={'fontFamily': 'Arial', 'fontSize': 14}),
        dcc.Dropdown(
            id='metadata-dropdown',
            options=dropdown_options,
            value=candidate_columns[0] if candidate_columns else None,
            clearable=False,
            style={'fontFamily': 'Arial', 'fontSize': 14}
        )
    ], style={'width': '50%', 'margin': 'auto', 'padding': '20px 0'}),
    dcc.Graph(id='histogram-graph')
])

# Callback to update the stacked histogram when a dropdown option is selected.
@app.callback(
    Output('histogram-graph', 'figure'),
    [Input('metadata-dropdown', 'value')]
)
def update_graph(selected_column):
    traces = []
    if selected_column is None:
        # If no metadata column is selected, simply show the overall histogram.
        counts, _ = np.histogram(score_data, bins=bin_edges, density=False)
        traces.append(go.Bar(
            x=bin_centers,
            y=counts,
            width=bin_width * 0.9,
            marker=dict(
                color=color_palette[0],
                line=dict(width=0)  # Remove border lines
            ),
            hoverinfo='x+y',
            name=''
        ))
    else:
        # Get unique values in the selected metadata column (sorted for consistency).
        unique_vals = sorted(df[selected_column].dropna().unique())
        # For each unique category, filter Score_SMAPE data and compute histogram counts.
        for i, val in enumerate(unique_vals):
            subset = df[df[selected_column] == val]['Score_SMAPE'].dropna().values
            counts, _ = np.histogram(subset, bins=bin_edges, density=False)
            # Only add a trace if there is at least one count in any bin.
            if np.sum(counts) == 0:
                continue
            traces.append(go.Bar(
                x=bin_centers,
                y=counts,
                width=bin_width * 0.9,
                name=str(val),
                marker=dict(
                    color=color_palette[i % len(color_palette)],
                    line=dict(width=0)  # Remove border lines
                ),
                hoverinfo='x+y'
            ))
    
    # Create the figure with stacked bar mode.
    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode='stack',
        xaxis_title='Score',
        yaxis_title='Count',
        template='simple_white',
        hovermode='x unified',
        font=dict(
            family="Arial",
            size=14,
            color="black"
        ),
        margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(title=clean_column_name(selected_column)) if selected_column else {}
    )
    
    # Update x-axis: ensure a thick black line.
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
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
