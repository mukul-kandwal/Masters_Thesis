import json
import os
from pprint import pprint # Pretty-print complex data structures
import gc  # Garbage collection to manage memory
from itertools import combinations  # Generate combinations of distinct seasons
import pandas as pd  # For handling dataframes and performing data manipulation
from datetime import datetime  # If needed for additional date handling
import numpy as np
import sys
PredictionsMetaData_dict = {}
output_directory = sys.argv[1]
# Check if sys.argv[4] is provided, default to 'abs' if not
if len(sys.argv) > 4:
    relative_first = sys.argv[4].lower() == 'rel'  # True if 'rel', False otherwise
else:
    relative_first = False  # Default to absolute differentiation

# Function to save each row
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
    pprint(data)
    with open(f"{filepath}", "a") as f:
        f.write(json.dumps(data) + "\n")

if len(sys.argv) > 5:
    Horizon = int(sys.argv[5])
else:
    Horizon = 1

def prepare_panel_data(time_series_list, trunc_len, fill_value):
    max_len = 34
    data = []
    target = []
    horizon = Horizon
    # Iterate over each time series (each season)
    for series in time_series_list:
        flucount = series.FluCount['Count']
        for i in range(horizon, len(flucount)):  # Adjust the starting index to accommodate the horizon
            # Input: past weeks up to week i-horizon
            data.append(flucount[:i - horizon + 1])
            # Output: the value at week i (horizon weeks ahead)
            target.append(flucount[i])
    
    # Padding each input sequence with zeros to make lengths equal (to handle variable-length series)
    data = [np.pad(x, (max_len - len(x), 0), 'constant', constant_values=fill_value)[-trunc_len:] for x in data]
    
    return np.array(data), np.array(target)

def score(Predictions, True_Values):
  score_sum = 0
  for a, b in zip(Predictions, True_Values):
    diff = a - b
    score_sum += (diff*diff)
  return score_sum/len(Predictions)

def predict_zero(data_set, test_num, max_length, pad_value):
        
    # Extract distinct seasons from the list of objects
    distinct_seasons = set(obj.Metadata.get("Season") for obj in data_set if "Season" in obj.Metadata)
    
    # Generate all combinations of the distinct seasons of length test_num
    training_season_sets = list(combinations(distinct_seasons, test_num))
    # pprint(training_season_sets)
    # pprint(len(training_season_sets))
    net_score = 0
    for training_season in training_season_sets:
      net_score += zero_script(data_set, training_season, max_length, pad_value)
      
    return net_score/len(training_season_sets)

def zero_script(data_set, test_seasons, max_length, pad_value):
    training_set = []
    test_set = []

    # Loop through each object in the list
    for obj in data_set:
        # Check if the object's Metadata has the "Season" key and it matches the training_seasons
        if obj.Metadata.get("Season") in test_seasons:
            test_set.append(obj)
        else:
            training_set.append(obj)
    
    n = 0
    test_score = 0
    Predictions = []
    
    for series in test_set:
        # Prepare the test data
        prediction_dates = series.FluCount["Week Ending Date"].tolist()
        season_week = series.FluCount["Season Week"].tolist()
        prediction_dates = [ts.strftime('%Y-%m-%d') for ts in prediction_dates]
        X_test, y_test = prepare_panel_data([series], max_length, pad_value)
        prediction_values = [0]  # Start with an initial prediction value
        
        # Use GPU-accelerated prediction
        for x_test in X_test:
            prediction_values.append(0)
        
        # Collect predictions for the current series
        PredictionsMetaData_dict = {
            'Season': series.Metadata.get("Season"),
            'Region': series.Metadata.get("Region"),
            'County': series.Metadata.get("County"),
            'Disease': series.Metadata.get("Disease")
        }
        
        # Calculate test score using a custom scoring function
        test_score += score(prediction_values[1:], y_test)
        grid_search_dict = {
          
        }
        # pprint({**PredictionsMetaData_dict, **grid_search_dict, **METADATA})
        # pprint(f"{prediction_values}, {y_test.tolist()}, {season_week}")
        save_rowwise_data(prediction_values, y_test.tolist(), season_week, {**grid_search_dict, **METADATA, **PredictionsMetaData_dict}, ["Season", "Dataset", "Model"])
        n += 1

    # Clear memory after each batch to prevent memory overflow
    del X_test, y_test, prediction_values
    gc.collect()  # Explicit garbage collection
    
    return test_score/n

def predict_last_value(data_set, test_num, max_length, pad_value):
        
    # Extract distinct seasons from the list of objects
    distinct_seasons = set(obj.Metadata.get("Season") for obj in data_set if "Season" in obj.Metadata)
    
    # Generate all combinations of the distinct seasons of length test_num
    training_season_sets = list(combinations(distinct_seasons, test_num))
    # pprint(training_season_sets)
    # pprint(len(training_season_sets))
    net_score = 0
    for training_season in training_season_sets:
      net_score += last_value_script(data_set, training_season, max_length, pad_value)
      
    return net_score/len(training_season_sets)

def last_value_script(data_set, test_seasons, max_length, pad_value):
    training_set = []
    test_set = []

    # Loop through each object in the list
    for obj in data_set:
        # Check if the object's Metadata has the "Season" key and it matches the training_seasons
        if obj.Metadata.get("Season") in test_seasons:
            test_set.append(obj)
        else:
            training_set.append(obj)
    
    n = 0
    test_score = 0
    Predictions = []
    
    for series in test_set:
        # Prepare the test data
        prediction_dates = series.FluCount["Week Ending Date"].tolist()
        season_week = series.FluCount["Season Week"].tolist()
        X_test, y_test = prepare_panel_data([series], max_length, pad_value)
        prediction_values = [0]  # Start with an initial prediction value
        
        # Use GPU-accelerated prediction
        for x_test in X_test:
            prediction_values.append(int(next((x for x in reversed(x_test) if x != 0), 0)))
        
        # copy below this line __________________________________
        # Collect predictions for the current series
        PredictionsMetaData_dict = {
            'Season': series.Metadata.get("Season"),
            'Region': series.Metadata.get("Region"),
            'County': series.Metadata.get("County"),
            'Disease': series.Metadata.get("Disease")
        }
        
        # Calculate test score using a custom scoring function
        test_score += score(prediction_values[1:], y_test)
        grid_search_dict = {
        }
        
        save_rowwise_data(prediction_values, y_test.tolist(), season_week, {**grid_search_dict, **METADATA, **PredictionsMetaData_dict}, ["Season", "Dataset", "Model"])
        n += 1

    # Clear memory after each batch to prevent memory overflow
    del X_test, y_test, prediction_values
    gc.collect()  # Explicit garbage collection
    
    return test_score/n

def predict_running_avg(data_set, test_num, max_length, pad_value):
        
    # Extract distinct seasons from the list of objects
    distinct_seasons = set(obj.Metadata.get("Season") for obj in data_set if "Season" in obj.Metadata)
    
    # Generate all combinations of the distinct seasons of length test_num
    training_season_sets = list(combinations(distinct_seasons, test_num))
    # pprint(training_season_sets)
    # pprint(len(training_season_sets))
    net_score = 0
    for training_season in training_season_sets:
      net_score += running_avg_script(data_set, training_season, max_length, pad_value)
      
    return net_score/len(training_season_sets)

def running_avg_script(data_set, test_seasons, max_length, pad_value):
    training_set = []
    test_set = []

    # Loop through each object in the list
    for obj in data_set:
        # Check if the object's Metadata has the "Season" key and it matches the training_seasons
        if obj.Metadata.get("Season") in test_seasons:
            test_set.append(obj)
        else:
            training_set.append(obj)
    
    n = 0
    test_score = 0
    Predictions = []
    
    for series in test_set:
        # Prepare the test data
        prediction_dates = series.FluCount["Week Ending Date"].tolist()
        season_week = series.FluCount["Season Week"].tolist()
        X_test, y_test = prepare_panel_data([series], max_length, pad_value)
        prediction_values = [0]  # Start with an initial prediction value
        
        # Use GPU-accelerated prediction
        for x_test in X_test:
            """Predicts the next value based on the running average of non-zero, non-negative values."""
            valid_values = [x for x in x_test if x > 0]
            
            if len(valid_values) > 0:
                # Calculate the average of valid values
                running_average = sum(valid_values) / len(valid_values)
                prediction_values.append(running_average)
            else:
                # If no valid values, predict 0
                prediction_values.append(0)

        
        # Collect predictions for the current series
        PredictionsMetaData_dict = {
            'Season': series.Metadata.get("Season"),
            'Region': series.Metadata.get("Region"),
            'County': series.Metadata.get("County"),
            'Disease': series.Metadata.get("Disease")
        }
        
        # Calculate test score using a custom scoring function
        test_score += score(prediction_values[1:], y_test)
        grid_search_dict = {
        }
        
        save_rowwise_data(prediction_values, y_test.tolist(), season_week, {**grid_search_dict, **METADATA, **PredictionsMetaData_dict}, ["Season", "Dataset", "Model"])
        n += 1

    # Clear memory after each batch to prevent memory overflow
    del X_test, y_test, prediction_values
    gc.collect()  # Explicit garbage collection
    
    return test_score/n

def predict_constant_change(data_set, test_num, max_length, pad_value):
        
    # Extract distinct seasons from the list of objects
    distinct_seasons = set(obj.Metadata.get("Season") for obj in data_set if "Season" in obj.Metadata)
    
    # Generate all combinations of the distinct seasons of length test_num
    training_season_sets = list(combinations(distinct_seasons, test_num))
    # pprint(training_season_sets)
    # pprint(len(training_season_sets))
    net_score = 0
    for training_season in training_season_sets:
      net_score += constant_change_script(data_set, training_season, max_length, pad_value)
      
    return net_score/len(training_season_sets)

def constant_change_script(data_set, test_seasons, max_length, pad_value):
    training_set = []
    test_set = []

    # Loop through each object in the list
    for obj in data_set:
        # Check if the object's Metadata has the "Season" key and it matches the training_seasons
        if obj.Metadata.get("Season") in test_seasons:
            test_set.append(obj)
        else:
            training_set.append(obj)
    
    n = 0
    test_score = 0
    Predictions = []
    
    for series in test_set:
        # Prepare the test data
        prediction_dates = series.FluCount["Week Ending Date"].tolist()
        season_week = series.FluCount["Season Week"].tolist()
        X_test, y_test = prepare_panel_data([series], max_length, pad_value)
        prediction_values = [0]  # Start with an initial prediction value
        
        # Use GPU-accelerated prediction
        for x_test in X_test:
            non_zero_values = [x for x in x_test if x > 0]
            
            if len(non_zero_values) >= 2:
                # Calculate the change between the last two values
                delta = non_zero_values[-1] - non_zero_values[-2]
                prediction_values.append(int(non_zero_values[-1] + delta))
            elif len(non_zero_values) == 1:
                # If there is only one non-zero value, return it as the next prediction
                prediction_values.append(int(non_zero_values[-1]))
            else:
                # If no non-zero values, predict 0
                prediction_values.append(0)

        
        # Collect predictions for the current series
        PredictionsMetaData_dict = {
            'Season': series.Metadata.get("Season"),
            'Region': series.Metadata.get("Region"),
            'County': series.Metadata.get("County"),
            'Disease': series.Metadata.get("Disease")
        }
        
        # Calculate test score using a custom scoring function
        test_score += score(prediction_values[1:], y_test)
        grid_search_dict = {
        }
        
        save_rowwise_data(prediction_values, y_test.tolist(), season_week, {**grid_search_dict, **METADATA, **PredictionsMetaData_dict}, ["Season", "Dataset", "Model"])
        n += 1

    # Clear memory after each batch to prevent memory overflow
    del X_test, y_test, prediction_values
    gc.collect()  # Explicit garbage collection
    
    return test_score/n

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
    pprint(self.Metadata)
    pprint(self.FluCount)
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

written_metadata = extract_metadata_from_jsonl(sys.argv[1])  # Pass output_directory as the first argument

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


#METADATA = {"Dataset": "FullAgnostic", "Model": "Baseline_Zero", "Differenced": diff}
#FA_B0 = predict_zero(FullAgnostic_Tseries, 1, 5, 0)

#METADATA = {"Dataset": "RegionCountySeperated", "Model": "Baseline_Zero"}
#RCS_B0 = predict_zero(RegionCountySeperated_Tseries, 1, 5, 0)

#METADATA = {"Dataset": "FullAgnostic", "Model": "Baseline_LastValue", "Differenced": diff}
#FA_BLast = predict_last_value(FullAgnostic_Tseries, 1, 5, 0)

#METADATA = {"Dataset": "RegionCountySeperated", "Model": "Baseline_LastValue"}
#RCS_BLast = predict_last_value(RegionCountySeperated_Tseries, 1, 5, 0)

METADATA = {"Dataset": "FullAgnostic", "Model": "Baseline_RunAVG", "Differenced": diff, "Horizon": Horizon}
FA_BRA = predict_running_avg(FullAgnostic_Tseries, 1, 5, 0)

#METADATA = {"Dataset": "RegionCountySeperated", "Model": "Baseline_RunAVG"}
#RCS_BRA = predict_running_avg(RegionCountySeperated_Tseries, 1, 5, 0)

#METADATA = {"Dataset": "FullAgnostic", "Model": "Baseline_ConstChange", "Differenced": diff}
#FA_BLastChange = predict_constant_change(FullAgnostic_Tseries, 1, 5, 0)

#METADATA = {"Dataset": "RegionCountySeperated", "Model": "Baseline_ConstChange"}
#RCS_BLastChange = predict_constant_change(RegionCountySeperated_Tseries, 1, 5, 0)