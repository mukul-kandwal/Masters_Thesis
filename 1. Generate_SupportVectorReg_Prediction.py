import json
import os
from pprint import pprint # Pretty-print complex data structures
import gc  # Garbage collection to manage memory
from itertools import combinations  # Generate combinations of distinct seasons
from itertools import product
import pandas as pd  # For handling dataframes and performing data manipulation
from datetime import datetime  # If needed for additional date handling
import numpy as np
from sklearn.linear_model import SGDRegressor
from tqdm import tqdm
import sys
# Check if sys.argv[4] is provided, default to 'abs' if not
if len(sys.argv) > 4:
    relative_first = sys.argv[4].lower() == 'rel'  # True if 'rel', False otherwise
else:
    relative_first = False  # Default to absolute differentiation

output_directory = sys.argv[1]

# Function to save each row

def generate_combinations(param_grid):
    # Convert each list of lists to list of tuples temporarily
    temp_grid = {key: [tuple(value) if isinstance(value, list) else value for value in values] 
                 for key, values in param_grid.items()}
    
    # Generate combinations using product
    combinations = list(product(*temp_grid.values()))
    
    # Revert tuples back to lists for list-of-lists parameters
    result = []
    for combination in combinations:
        # Use zip to pair up keys with each value in the combination
        combo_dict = {}
        for key, value in zip(param_grid.keys(), combination):
            # Revert tuples to lists where applicable
            combo_dict[key] = list(value) if isinstance(value, tuple) and key == 'hidden_layer_sizes' else value
        result.append(combo_dict)
    return result

def flatten_json(json_obj, parent_key='', sep='_'):
    """
    Recursively flattens a JSON object, handling nested dictionaries.
    Column names will reflect the full path of keys, concatenated with underscores.
    """
    items = {}
    for k, v in json_obj.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def read_json_files_to_dataframe(folder_path):
    # Example usage:
    # folder_path = 'path_to_your_json_folder'
    # df = read_json_files_to_dataframe(folder_path)
    # print(df)
    
    """
    Reads a folder of JSON files into a pandas DataFrame, flattening nested dictionaries,
    and handling different keys across files (filling with NaN where necessary).
    """
    all_data = []
    
    # Loop over each file in the folder
    for filename in tqdm(os.listdir(folder_path), desc="Processing JSON files"):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                flattened_data = flatten_json(json_data)
                flattened_data.update({"filename": os.path.basename(file_path)})
                # flattened_data.update("filename": os.path.basename(filepath))
                all_data.append(flattened_data)
    
    # Create DataFrame from the list of flattened JSONs
    df = pd.DataFrame(all_data)
    
    # Fill missing keys with NaN
    df = df.fillna(pd.NA)  # or pd.NA if you prefer true NaN handling
    
    return df

def extract_metadata_from_jsonl(folder_path):
    metadata_list = []
    
    for filename in tqdm(os.listdir(folder_path), desc="Processing JSON files"):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    # Check if 'metadata' key exists in the current line
                    if 'metadata' in data:
                        metadata_list.append(data['metadata'])
    
    return metadata_list

def save_rowwise_data(forecast_values, actual_values, season_week, metadata, keys_for_filename):
    # Get the current timestamp (formatted to the minute)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    
    # Fetch the values from the dictionary to include in the file name
    values_for_filename = [str(metadata.get(key, f"default_{key}")) for key in keys_for_filename]
    
    # Create a safe file name by joining the selected values with underscores
    filename = f"{'_'.join(values_for_filename)}_{timestamp}.jsonl".replace(" ", "_")
    
    # Build the full path for the output file
    filepath = f"{output_directory}/{filename}"
    
    data = {
        'forecast_values': forecast_values,
        'actual_values': actual_values,
        'season_week': season_week,
        'metadata': metadata
    }
    with open(f"{filepath}", "a") as f:
        f.write(json.dumps(data) + "\n")

def prepare_panel_data(time_series_list, trunc_len, fill_value):
    max_len = 34
    data = []
    target = []
    
    # Iterate over each time series (each season)
    for series in time_series_list:
        flucount = series.FluCount['Count']
        for i in range(1, len(flucount)):
            # Input: past weeks up to week i-1
            data.append(flucount[:i])
            # Output: the value at week i (next value)
            target.append(flucount[i])
    
    # Padding each input sequence with zeros to make lengths equal (to handle variable-length series)
    # max_len = max(len(series.FluCount) for series in time_series_list)
    data = [np.pad(x, (max_len - len(x), 0), 'constant', constant_values=fill_value)[-trunc_len:] for x in data]
    # print("-------------------------------------------------------------------------------------")
    # pprint(np.array(data))
    # pprint(np.array(target))
    return np.array(data), np.array(target)

def score(Predictions, True_Values):
#   score_sum = 0
#   for a, b in zip(Predictions, True_Values):
#     diff = a - b
#     score_sum += (diff*diff)
#   return score_sum/len(Predictions)
    return 0.1

class Flu_Timeseries():
  def __init__(self, df):
        # Set the FluCount DataFrame with selected columns
        self.FluCount = df[["Week Ending Date", "Season Week", "Count"]]
        
        # Initialize the Metadata dictionary with default "Agg." values
        self.Metadata = {
            "Region": "Agg.",
            "County": "Agg.",
            "Season": "Agg.",
            "Disease": "Agg.",
            "Differenced": [0, "None"],
        }
        
        # Update Metadata values if the corresponding columns exist in the DataFrame
        for key in self.Metadata.keys():
            if key in df.columns:
                self.Metadata[key] = df[key].iloc[0]  # Use the first row's value for each key

  def difference(self, relative=False):
    """
    Calculates the difference in the 'Count' column. Can compute either absolute or relative differences.
    
    Parameters:
    relative (bool): If True, compute relative differences. Otherwise, compute absolute differences.
    
    Returns:
    self: Returns the updated object for method chaining.
    """
    if relative:
        # Calculate the relative difference in the "Count" column
        self.FluCount['Count'] = self.FluCount['Count'].pct_change()
    else:
        # Calculate the absolute difference in the "Count" column
        self.FluCount['Count'] = self.FluCount['Count'].diff()
    
    # Optionally drop rows with NaN values after differencing
    self.FluCount = self.FluCount.dropna().reset_index(drop=True)
    
    # Update Metadata
    diff_type = [self.Metadata["Differenced"][0] + 1, "Relative"] if relative else [self.Metadata["Differenced"][0] + 1, "Absolute"]
    self.Metadata["Differenced"] = diff_type
    
    # Return self to allow for method chaining if desired
    return self
  
  def display(self):
    print("-------------------------")
    print(self.Metadata)
    print(self.FluCount)
    print("-------------------------")

  def plot(self):
        """
        Plots a time series line graph with 'Count' on the y-axis and 'Week Ending Date' on the x-axis using Plotly.
        """
        # Ensure 'Week Ending Date' is in datetime format
        self.FluCount['Week Ending Date'] = pd.to_datetime(self.FluCount['Week Ending Date'])
        
        # Sort the DataFrame by 'Week Ending Date' if not already sorted
        self.FluCount = self.FluCount.sort_values('Week Ending Date')

        # Create the Plotly figure
        fig = go.Figure()

        # Add the line trace for 'Count' vs. 'Week Ending Date'
        fig.add_trace(go.Scatter(
            x=self.FluCount['Week Ending Date'],
            y=self.FluCount['Count'],
            mode='lines+markers',  # Line with markers at data points
            marker=dict(size=6),
            line=dict(color='blue'),
            name='Flu Count'
        ))

        # Add title and axis labels
        metadata_str = f"Region: {self.Metadata['Region']} | County: {self.Metadata['County']} | Season: {self.Metadata['Season']} | Disease: {self.Metadata['Disease']} | Differenced: {self.Metadata['Differenced']}"
        fig.update_layout(
            title=f"Flu Timeseries Data<br>{metadata_str}",
            xaxis_title="Week Ending Date",
            yaxis_title="Count",
            xaxis=dict(
                tickangle=-45,  # Rotate x-axis labels for better readability
                type="date",
                tickformat="%b %Y"  # Display dates in 'Month Year' format
            ),
            template="plotly_white",  # Clean plot background
            hovermode="x unified",  # Hover displays data for the same x-axis point
            height=600,  # Adjust height to scale better on different screens
            width=1000  # Adjust width to scale better on different screens
        )

        # Show the interactive plot
        fig.show()

def MakeTimeseries(df_list):
  obj_list  = []
  for df in df_list:
    obj_list.append(Flu_Timeseries(df))
  
  return obj_list

def split_dataframe_by_unique_values(df, column):

    unique_values = df[column].unique()
    split_dfs = []
    
    for value in unique_values:
        # Use the previously defined filter_dataframe function
        filtered_df = filter_dataframe(df, {column: value})
        split_dfs.append(filtered_df)
    
    return split_dfs

def split_and_merge_infer_columns(df, merge_columns):
    # List all categorical columns in the DataFrame
    categorical_columns = ['Region', 'County', 'Disease']
    
    # Infer the columns to split by (those not in merge_columns)
    split_columns = [col for col in categorical_columns if col not in merge_columns]
    
    if not split_columns:
      return [merge_rows_by_columns(df, merge_columns)]
    
    # Split the DataFrame by unique combinations of the inferred columns
    unique_combinations = df[split_columns].drop_duplicates()
    split_dfs = []
    
    for _, combo in unique_combinations.iterrows():
        # Filter the DataFrame by the current unique combination of split columns
        filtered_df = df.copy()
        for col in split_columns:
            filtered_df = filtered_df[filtered_df[col] == combo[col]]
        
        # Merge rows in the filtered DataFrame by the merge_columns
        merged_df = merge_rows_by_columns(filtered_df, merge_columns)
        
        # Append the merged DataFrame to the list
        split_dfs.append(merged_df.sort_values(by=['Week Ending Date'], kind="mergesort"))
    
    return split_dfs

def add_season_week(df):
    """
    Adds a 'Season Week' column to the DataFrame, calculated as ('CDC Week' + 13) % 52.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing a 'CDC Week' column.
    
    Returns:
    pandas.DataFrame: The DataFrame with the new 'Season Week' column.
    """
    # Ensure 'CDC Week' is an integer
    df['CDC Week'] = df['CDC Week'].astype(int)
    
    # Calculate 'Season Week'
    df['Season Week'] = (df['CDC Week'] + 13) % 52
    
    # Handling case where Season Week is 0 (it should be 52)
    df.loc[df['Season Week'] == 0, 'Season Week'] = 52
    
    df = df.drop(columns=['CDC Week'])
    return df

def filter_dataframe(df, filter_dict):
    # Ensure all keys in filter_dict exist in the DataFrame
    for col in filter_dict.keys():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
    
    # Apply the filter conditions
    filtered_df = df.copy()
    for col, value in filter_dict.items():
        filtered_df = filtered_df[filtered_df[col] == value]
    
    return filtered_df

def merge_rows_by_columns(df, cols):
    '''
    This function takes in dataframe and a list of column names.
    After checking if the columns provided are present in the data frame,
    it groups the dataframes by those columns, summing up the 'Count" column values"
    '''
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
    grouped_df = df.groupby([col for col in df.columns if col not in cols + ['Count']], as_index=False)['Count'].sum()
    return grouped_df

def SGDRegressor_script(data_set, test_seasons, Hyperparams):
    training_set, test_set = [], []
    for obj in data_set:
        if obj.Metadata.get("Season") in test_seasons:
            test_set.append(obj)
        else:
            training_set.append(obj)

    series = test_set[0]
    prediction_dates = [ts.strftime('%Y-%m-%d') for ts in series.FluCount["Week Ending Date"].tolist()]
    season_week = series.FluCount["Season Week"].tolist()

    PredictionsMetaData_dict = {
        'Model': "SGD Regressor",
        'Season': series.Metadata.get("Season"),
        'Region': series.Metadata.get("Region"),
        'County': series.Metadata.get("County"),
        'Disease': series.Metadata.get("Disease"),
    }

    if {**Hyperparams, **METADATA, **PredictionsMetaData_dict} in written_metadata:
        print("Skipping!")
        return 0
    else:
        print("Not Skipping!")

    X, y = prepare_panel_data(training_set, Hyperparams["max_length"], Hyperparams["pad_value"])

    model = SGDRegressor(
    alpha=Hyperparams["alpha"],             # Regularization strength
    max_iter=Hyperparams["max_iter"],       # Maximum number of iterations
    tol=Hyperparams["tol"],                 # Stopping criterion tolerance
    learning_rate=Hyperparams["learning_rate"],  # Learning rate schedule
    eta0=Hyperparams.get("eta0", 0.01),     # Initial learning rate
    random_state=48)
    model.fit(X, y)

    X_test, y_test = prepare_panel_data([series], Hyperparams["max_length"], Hyperparams["pad_value"])
    prediction_values = [float(model.predict(np.array([x_test]))[0]) for x_test in X_test]

    test_score = score(prediction_values, y_test)
    save_rowwise_data(prediction_values, y_test.tolist(), season_week, {**Hyperparams, **METADATA, **PredictionsMetaData_dict}, ["Model", "Dataset", "Season"])

    del X, y, X_test, y_test, prediction_values
    gc.collect()

    return test_score

def predict_SupportVector(data_set, test_num, Hyperparams):
    
    # Extract distinct seasons from the list of objects
    distinct_seasons = set(obj.Metadata.get("Season") for obj in data_set if "Season" in obj.Metadata)
    
    # Generate all combinations of the distinct seasons of length test_num
    training_season_sets = list(combinations(distinct_seasons, test_num))
    # pprint(training_season_sets)
    # pprint(len(training_season_sets))
    net_score = 0
    for training_season in training_season_sets:
      print("Going to script")
      net_score += SGDRegressor_script(data_set, training_season, Hyperparams)
      
    return net_score/len(training_season_sets)

def grid_search_for_SVR(param_grid, data_set):
    # Create list of all hyperparameter combinations
    param_combinations = generate_combinations(param_grid)

    # Iterate through all combinations and train models
    for i, param_values in enumerate(param_combinations):
        # print(params)
        print(f"Training model {i + 1}/{len(param_combinations)}")
        # Train model with the current hyperparameters
        score = predict_SupportVector(data_set, 1, param_values)
        

print("Reading folder")
written_metadata = extract_metadata_from_jsonl(output_directory)

# Read in the data from the raw data CSV and ensure the dates are read as type date
Raw_Data = pd.read_csv(sys.argv[2])
Raw_Data = Raw_Data[~Raw_Data['Season'].isin(['2021-2022', '2022-2023'])]
Raw_Data['Week Ending Date'] = pd.to_datetime(Raw_Data['Week Ending Date'], format='%m/%d/%Y')
Raw_Data = add_season_week(Raw_Data)

# Remove the irrelevant columns and split the data by season
Raw_Data = Raw_Data.drop(columns=['FIPS', 'County Centroid'])
Year_Seperated_Data = split_dataframe_by_unique_values(Raw_Data, 'Season')

# Generate time series data
FullAgnostic = []
for Time_Series in Year_Seperated_Data:
    FullAgnostic.extend(split_and_merge_infer_columns(Time_Series, ['Disease', 'County', 'Region']))
FullAgnostic_Tseries = MakeTimeseries(FullAgnostic)

# Dynamic .difference() Logic
num_differences = int(sys.argv[3])  # Number of .difference() calls

for TimeSeries in FullAgnostic_Tseries:
    # Perform the first .difference call based on the 'rel' argument
    if num_differences > 0 and relative_first:
        TimeSeries.difference(True)
        num_differences -= 1
    
    # Perform the remaining .difference calls
    for _ in range(num_differences):
        TimeSeries.difference()

# Extract the "Differenced" metadata
diff = FullAgnostic_Tseries[0].Metadata["Differenced"][0]

# Define metadata and perform grid search
METADATA = {
    "Training_Restriction": "None",
    "Dataset": "FullAgnostic_Tseries",
    "SpatialFeature": False,
    "SeasonFeature": False,
    "Differenced": diff,
}

grid_search_dict = {
    "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1],  # Regularization strength (inverse of C)
    "max_iter": [500, 1000, 2000],
    "tol": [1e-3, 1e-4],  # Stopping tolerance
    "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
    "eta0": [0.001, 0.01, 0.1],  # Initial learning rate for 'constant' or 'adaptive'
    'max_length': [5, 10, 20, 34],
    'pad_value': [-1, 0]
}

grid_search_for_SVR(grid_search_dict, FullAgnostic_Tseries)

# METADATA = {"Training_Restriction": "None", "Dataset": "RegionCountySeperated_Tseries", "SpatialFeature": False, "SeasonFeature": False}
# grid_search_for_XGB(grid_search, RegionCountySeperated_Tseries)



# FA_F0_P34_XGBR = Run_XGBRegressor(FullAgnostic_Tseries, 1, 34, 0)
# FA_F1_P34_XGBR = Run_XGBRegressor(FullAgnostic_Tseries, 1, 34, -1)
# FA_F0_P5_XGBR = Run_XGBRegressor(FullAgnostic_Tseries, 1, 5, 0)
# FA_F1_P5_XGBR = Run_XGBRegressor(FullAgnostic_Tseries, 1, 5, -1)
# RCS_F0_P34_XGBR = Run_XGBRegressor(RegionCountySeperated_Tseries, 1, 34, 0)
# RCS_F1_P34_XGBR = Run_XGBRegressor(RegionCountySeperated_Tseries, 1, 34, -1)
# RCS_F0_P5_XGBR = Run_XGBRegressor(RegionCountySeperated_Tseries, 1, 5, 0)
# RCS_F1_P5_XGBR = Run_XGBRegressor(RegionCountySeperated_Tseries, 1, 5, -1)

