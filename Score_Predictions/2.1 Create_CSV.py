import os
import sys
import pandas as pd
import json
from tqdm import tqdm

def flatten_json(json_obj, parent_key='', sep='_'):
    """Recursively flattens a JSON object with nested dictionaries."""
    items = {}
    for k, v in json_obj.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def read_json_files_to_dataframe(folder_path, dropped_columns = None):
    # Example usage:
    # folder_path = 'path_to_your_json_folder'
    # df = read_json_files_to_dataframe(folder_path)
    # print(df)
    
    """
    Reads a folder of JSON files into a pandas DataFrame, flattening nested dictionaries,
    and handling different keys across files (filling with NaN where necessary).
    """
    all_data = pd.DataFrame()
    
    # Loop over each file in the folder
    for filename in tqdm(os.listdir(folder_path), desc="Processing JSON files"):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                try:
                  json_data = json.load(file)
                except Exception as e:
                  print("An error occurred:", e)
                flattened_data = flatten_json(json_data)
                if dropped_columns:   
                  flattened_data = {k: v for k, v in flattened_data.items() if k not in dropped_columns}
                  pass
                flattened_data.update({"filename": os.path.basename(file_path)})
                # flattened_data.update("filename": os.path.basename(filepath))
                all_data = pd.concat([all_data, pd.DataFrame([flattened_data])], ignore_index = True)
    
    # Create DataFrame from the list of flattened JSONs
    # df = pd.DataFrame(all_data)
    
    # Fill missing keys with NaN
    df = all_data.fillna(pd.NA)  # or pd.NA if you prefer true NaN handling
    
    return df

def split_dataframe_to_csvs(df, output_folder, split_columns):
    """
    Splits a DataFrame into multiple CSVs based on unique combinations in specified columns
    and saves them to the specified output folder.

    Parameters:
    - df (pd.DataFrame): The DataFrame to split.
    - output_folder (str): The path to the folder where CSV files should be saved.
    """
    unique_combinations = df.groupby(
        split_columns
    )
    
    for keys, subset in unique_combinations:
        # Convert tuple of unique values to a string joined by underscores
        filename = "_".join([str(k) for k in keys]) + ".csv"
        # Full path to save the CSV
        file_path = os.path.join(output_folder, filename)
        # Save the subset DataFrame to CSV
        subset.to_csv(file_path, index=False)
        print(f"Saved {file_path}")


if __name__ == "__main__":
    # Ensure the script is run with the correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_folder_path> <output_folder_path>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    # Process JSONL files
    df = read_json_files_to_dataframe(input_folder)
    
    # Define potential columns to split by
    potential_split_columns = [
        'metadata_Differenced',
        'metadata_SpatialFeature', 
        'metadata_SeasonFeature', 
        'metadata_Model'
    ]
    
    # Infer split_columns dynamically as those present in the DataFrame
    split_columns = [col for col in potential_split_columns if col in df.columns]
    
    # Split and save DataFrame by unique combinations in specified columns
    split_dataframe_to_csvs(df, output_folder, split_columns)