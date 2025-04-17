from pprint import pprint
import os
import sys
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingRegressor  # For surrogate model
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

def display_or_save_plot(fig, folder_path=None, continuous_col=None, discrete_col=None):
    """
    Displays the plot or saves it to the specified folder path as an HTML file.

    Parameters:
    - fig (plotly.graph_objects.Figure): The plot object to display or save.
    - folder_path (str, optional): The directory path to save the plot. The filename
      will be generated based on the column details.
    - discrete_col (str, optional): The name of the discrete column used in the plot.
    - continuous_col (str, optional): The name of the continuous column used in the plot.
    """
    first_trace_name = fig.data[0].name
    # Extract discrete and continuous column names if not provided
    if not discrete_col:
        discrete_col = first_trace_name.split('=')[0].strip()
        
    if not continuous_col:
        continuous_col = fig.layout.xaxis.title.text

    # Generate filename based on columns and path
    if folder_path:
        filename = f"{discrete_col}_{continuous_col}.html"
        full_path = os.path.join(folder_path, filename)

        # Save as HTML file
        fig.write_html(full_path)
        print(f"Plot saved as HTML to {full_path}")
    else:
        fig.show()

def filter_by_metadata(df, metadata_key, metadata_value):
    return df[df['metadata'].apply(lambda x: x.get(metadata_key) == metadata_value)]

def subset_data(df, metadata_filters, columns_to_return):
    # Generalized function to filter by multiple metadata keys and select specific columns
    # Apply each filter condition on the metadata
    for key, value in metadata_filters.items():
        df = df[df['metadata'].apply(lambda x: x.get(key) == value)]
    
    # Return specified columns
    return df[columns_to_return]

def extract_metadata_from_jsonl(folder_path):
    metadata_list = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    # Check if 'metadata' key exists in the current line
                    if 'metadata' in data:
                        metadata_list.append(data['metadata'])
    
    return metadata_list

def aggregate_scores(df, classifier_cols, agg_method='mean'):
    if not classifier_cols:
        raise ValueError("No group keys passed!")

    # Dynamically identify score columns
    score_cols = [col for col in df.columns if col.startswith("Score_")]

    if not score_cols:
        raise ValueError("No score columns found for aggregation!")

    # Group and aggregate scores
    aggregated_df = (
        df.groupby(classifier_cols)[score_cols]
        .agg(agg_method)
        .reset_index()
    )
    return aggregated_df

def plot_score_distributions_violin(df, param_cols, score):
    """
    Plots score distributions as violin plots with a mean line for each parameter.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - param_cols (list of str): List of column names for parameters to plot against the score.
    """
    for col in param_cols:
        fig = px.violin(df, x=col, y=score, box=True, points="all",
                        title=f'Score Distribution by {col} (Violin Plot with Mean Line)')

        # Calculate mean score for each parameter value
        mean_line = df.groupby(col)[score].mean().reset_index()

        # Add mean line to the plot
        fig.add_trace(go.Scatter(
            x=mean_line[col],
            y=mean_line[score],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Mean Score'
        ))

    return fig

def plot_partial_dependence(df, param_cols, score):
    '''
    plot_partial_dependence(df, param_columns)
    '''
    X = df[param_cols]
    y = df[score]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a surrogate model
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    # Create partial dependence data
    partial_deps = {}
    for col in param_cols:
        partial_deps[col] = np.linspace(X[col].min(), X[col].max(), 100)

    fig = go.Figure()
    for col in param_cols:
        avg_scores = [model.predict(X.assign(**{col: val})) for val in partial_deps[col]]
        fig.add_trace(go.Scatter(x=partial_deps[col], y=avg_scores, mode='lines', name=col))

    fig.update_layout(title='Partial Dependence Plots', xaxis_title='Parameter Value', yaxis_title='Predicted Score')
    return fig

def quantile_analysis(df, param_cols, score):
    '''
    quantile_analysis(df, param_columns)
    '''
    quantiles = df[score].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print("Score Quantiles:\n", quantiles)

    for col in param_cols:
        low_val = df.loc[df[score] <= quantiles[0.05]][col].min()
        high_val = df.loc[df[score] <= quantiles[0.05]][col].max()
        print(f"For {col}, low value in bottom 5%: {low_val}, high value: {high_val}")

def plot_marginal_gains(df, param_cols, score):
    '''
    plot_marginal_gains(df, param_columns)
    '''
    fig = go.Figure()

    for col in param_cols:
        fig.add_trace(go.Scatter(
            x=df[col],
            y=df[score].diff(),
            mode='markers',
            name=col
        ))

    fig.update_layout(title='Marginal Gains Analysis', xaxis_title='Parameter Value', yaxis_title='Score Improvement')
    return fig

def find_min_row(df, score):
    """
    Finds the row with the minimum value in the specified column.

    Parameters:
    - df (pd.DataFrame): The DataFrame to search.
    - column_name (str): The column in which to find the minimum value.

    Returns:
    - pd.Series: The row with the minimum value in the specified column.
    """
    min_row = df.loc[df[score].idxmin()]
    return min_row

def get_least_value_rows(df, column_name, num_rows=1, ascending=True):
    """
    Get the top 'num_rows' rows with the least values in a specified column from a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to search.
    - column_name (str): The name of the column to evaluate.
    - num_rows (int): The number of rows to return. Default is 1.

    Returns:
    - pd.DataFrame: A DataFrame containing the top 'num_rows' rows with the least values.
    """
    # Ensure the specified column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Sort the DataFrame by the specified column in ascending order
    sorted_df = df.sort_values(by=column_name, ascending=ascending)
    
    # Return the top 'num_rows' rows
    return sorted_df.head(num_rows)

def plot_cumulative_frequency(df, discrete_col, continuous_col, ascending=True, interval=1):
    """
    Generate cumulative frequency plots for all unique values in a discrete column on the same graph,
    including reference dotted lines at specific percentages of the x-axis (0.1%, 1%, 5%, 25%, and 50%),
    and plot every n-th point for smoothing.

    Parameters:
    - df (pd.DataFrame): The DataFrame to analyze.
    - discrete_col (str): The name of the discrete column.
    - continuous_col (str): The name of the continuous column.
    - interval (int): The interval at which to plot points for smoothing.
    """
    # Ensure the specified columns exist in the DataFrame
    if discrete_col not in df.columns or continuous_col not in df.columns:
        raise ValueError(f"One or both columns '{discrete_col}' or '{continuous_col}' do not exist in the DataFrame.")
    
    # Sort the DataFrame by the continuous column
    sorted_df = df.sort_values(by=continuous_col, ascending=ascending)
    # Initialize a figure
    fig = go.Figure()

    # Get unique values from the discrete column
    unique_values = df[discrete_col].unique()
    num_unique_values = len(unique_values)
    reference_percent = 1 / num_unique_values  # Calculate reference percentage

    # Calculate and plot cumulative frequency for each unique value
    for value in unique_values:
        # Initialize the count of the unique value
        count_value = 0
        cumulative_counts = []
        
        total_rows = len(sorted_df)

        for i in range(total_rows):
            # Check if the current row's discrete value matches the unique value
            if sorted_df.iloc[i][discrete_col] == value:
                count_value += 1
            
            # Calculate cumulative percentage
            cumulative_percentage = count_value / (i + 1)
            cumulative_counts.append(cumulative_percentage)

        # Add line to the plot, selecting every n-th point
        x_values = list(range(1, total_rows + 1))
        fig.add_trace(go.Scatter(
            x=x_values[::interval],  # Plot every n-th point
            y=cumulative_counts[::interval],
            mode='lines',
            name=f'{discrete_col} = {value}'
        ))

    # Add a reference dotted line at the calculated percentage
    fig.add_shape(type="line",
                  x0=0, y0=reference_percent, x1=total_rows, y1=reference_percent,
                  line=dict(color="Red", dash="dash"),  # Dotted line
                  name="Reference Line")

    # Define percentages to mark along the x-axis
    percentages = [0.001, 0.01, 0.05, 0.25, 0.5]
    
    # Add dotted lines for each specified percentage
    for p in percentages:
        line_x = p * total_rows
        fig.add_shape(type="line",
                      x0=line_x, y0=0, x1=line_x, y1=1,
                      line=dict(color="Black", dash="dash", width=1),  # Thin black dotted line
                      name=f"{int(p * 100)}% Line")

    # Update layout
    fig.update_layout(
        title='Cumulative Frequency for Discrete Column',
        xaxis_title='Number of Rows',
        yaxis_title='Cumulative Percentage',
        yaxis=dict(tickformat='.0%'),  # Format y-axis as percentage
        showlegend=True
    )

    # Add a reference line annotation
    fig.add_annotation(
        x=total_rows,
        y=reference_percent,
        text=f'Reference Line: {reference_percent:.2%}',
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        bgcolor="lightgrey",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        font=dict(color="black")
    )

    return fig

def plot_cumulative_frequency_histogram(df, discrete_col, continuous_col, interval):
    """
    Generate cumulative frequency histograms for all unique values in a discrete column on the same graph.

    Parameters:
    - df (pd.DataFrame): The DataFrame to analyze.
    - discrete_col (str): The name of the discrete column.
    - continuous_col (str): The name of the continuous column.
    - interval (int): The number of rows to group for frequency calculation.
    """
    # Ensure the specified columns exist in the DataFrame
    if discrete_col not in df.columns or continuous_col not in df.columns:
        raise ValueError(f"One or both columns '{discrete_col}' or '{continuous_col}' do not exist in the DataFrame.")
    
    # Sort the DataFrame by the continuous column
    sorted_df = df.sort_values(by=continuous_col)

    # Initialize a figure
    fig = go.Figure()

    # Get unique values from the discrete column
    unique_values = df[discrete_col].unique()

    # Calculate and plot cumulative frequency for each unique value
    total_rows = len(sorted_df)

    for value in unique_values:
        # Initialize cumulative counts and the number of bins
        cumulative_counts = []
        count_value = 0

        for i in range(0, total_rows, interval):
            # Determine the end index for the current interval
            end_index = min(i + interval, total_rows)

            # Count occurrences of the unique value in the current interval
            count_value += (sorted_df[discrete_col].iloc[i:end_index] == value).sum()

            # Calculate cumulative percentage
            cumulative_percentage = count_value / (end_index)
            cumulative_counts.append(cumulative_percentage)

        # Add histogram to the plot
        fig.add_trace(go.Bar(
            x=list(range(1, len(cumulative_counts) + 1)),  # Bin numbers
            y=cumulative_counts,
            name=f'{discrete_col} = {value}',
            opacity=0.75
        ))

    # Update layout
    fig.update_layout(
        title='Cumulative Frequency Histogram for Discrete Column',
        xaxis_title='Bins (Every X Rows)',
        yaxis_title='Cumulative Percentage',
        yaxis=dict(tickformat='.0%'),  # Format y-axis as percentage
        showlegend=True
    )

    return fig

def print_and_save_best_scores(df, column_name, num_rows=5, ascending=True):
    """
    Prints the DataFrame rows with the least values in the specified column using pprint
    and saves the output to a text file.

    Parameters:
    - df (DataFrame): The DataFrame to search for the least value rows.
    - column_name (str): The name of the column to find the least values for.
    - num_rows (int): The number of rows to return with the least values. Default is 5.
    - ascending (bool): If True, sort in ascending order; otherwise, descending. Default is True.
    """
    # Get the DataFrame rows with the least values in the specified column
    best_scores_df = get_least_value_rows(df, column_name, num_rows=num_rows, ascending=ascending)
    
    # Convert the DataFrame to a dictionary format for pretty printing
    best_scores_dict = best_scores_df.to_dict(orient='records')
    
    # Print the output
    print(f"\nBest scores for {column_name} (Top {num_rows}, {'ascending' if ascending else 'descending'}):\n")
    print(best_scores_dict)
    
    # Save the output to a text file
    filename = f"best_scores_{column_name}.txt"
    with open(os.path.join(output_folder, filename), "w") as file:
        file.write(f"Best scores for {column_name} (Top {num_rows}, {'ascending' if ascending else 'descending'}):\n\n")
        print(best_scores_dict, file=file)
        print(f"Output saved to {filename}")


# Load all the files in the output folder
if len(sys.argv) != 3:
    print("Usage: python3 3. Analyze_Grid_Search.py input_path Output_folder")
    sys.exit(1)

# Get paths from command-line arguments
input_path = sys.argv[1]
output_folder = sys.argv[2]

df = pd.read_csv(input_path)
if 'metadata_Training_Restriction' in df.columns:
  df['metadata_Training_Restriction'].fillna("none", inplace=True)
print("FILE READ FULLY _________________________________________________")


excluded_columns = ['metadata_Training_Restriction', 'metadata_Dataset',
'metadata_SpatialFeature', 'metadata_SeasonFeature', 'metadata_Model']


hyperparams = [
    col for col in df.columns 
    if col.startswith("metadata_") 
    and col not in excluded_columns 
    and df[col].nunique() > 1
    and col != "metadata_Season"
]

print(f"Hyperparams being analyzed: {hyperparams}")

classifier_cols = [col for col in df.columns if col.startswith("metadata_")]
df_agg = aggregate_scores(df, classifier_cols)

print(df_agg.head())  # Check the first few rows
print(df_agg.columns)  # Verify column names
print(df_agg['Score_RMSE'].dtype)  # Ensure it is numeric
print(df_agg['Score_RMSE'].isna().sum())  # Count NaN values

print(df_agg['Score_RMSE'])    
print(find_min_row(df_agg, "Score_RMSE"))
print(find_min_row(df_agg, "Score_MAE"))
print(find_min_row(df_agg, "Score_MSE"))
print(find_min_row(df_agg, "Score_MAPE"))
print(find_min_row(df_agg, "Score_R-squared"))
print(find_min_row(df_agg, "Score_Global Forecast Deviance"))

# Calling the function for each score column
score_columns = ["Score_RMSE", "Score_MAE", "Score_MSE", "Score_MAPE", "Score_R-squared", "Score_Global Forecast Deviance"]
for column in score_columns:
    if column == "Score_R-squared":
      ascending = False
    else:
      ascending = True
    print_and_save_best_scores(df_agg, column, num_rows=10, ascending=ascending)

print(get_least_value_rows(df_agg, "Score_RMSE"))
print(get_least_value_rows(df_agg, "Score_MAE"))
print(get_least_value_rows(df_agg, "Score_MSE"))
print(get_least_value_rows(df_agg, "Score_MAPE"))
print(get_least_value_rows(df_agg, "Score_R-squared"))
print(get_least_value_rows(df_agg, "Score_Global Forecast Deviance"))

# display_or_save_plot(plot_score_distributions_violin(df, ['n_estimators', 'learning_rate', 'max_depth',
#        'subsample', 'colsample_bytree', 'gamma',
#        'reg_alpha', 'reg_lambda', 'max_length', 'pad_value'], "Score_RMSE"), input_folder)

# quantile_analysis(df, ['n_estimators', 'learning_rate', 'max_depth',
#        'subsample', 'colsample_bytree', 'gamma',
#        'reg_alpha', 'reg_lambda', 'max_length', 'pad_value'], "Score_RMSE")

# display_or_save_plot(plot_marginal_gains(df, ['n_estimators', 'learning_rate', 'max_depth',
#        'subsample', 'colsample_bytree', 'gamma',
#        'reg_alpha', 'reg_lambda', 'max_length', 'pad_value'], "Score_RMSE"), input_folder)
       
# display_or_save_plot(plot_partial_dependence(df_agg, ['metadata_alpha', 'metadata_learning_rate_init', 
#                              'metadata_max_iter', 'metadata_max_length'], "Score_RMSE"), input_folder)


# display_or_save_plot(plot_cumulative_frequency_histogram(df, "metadata_hidden_layer_sizes", "Score_RMSE", 10), input_folder)
hyperparams.append("metadata_Season")
for score_type in tqdm(score_columns, desc="Outer Loop", position=0):
    if column == "Score_R-squared":
      ascending = False
    else:
      ascending = True    
    for hyperparam in tqdm(hyperparams, desc="Inner Loop", position=1, leave=False):
        if hyperparam == "metadata_Season":
          skip_len = 2
        else:
          skip_len = 5
        display_or_save_plot(plot_cumulative_frequency(df, hyperparam, score_type, ascending, skip_len), output_folder, score_type)

