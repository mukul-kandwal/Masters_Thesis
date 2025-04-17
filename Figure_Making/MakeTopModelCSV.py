import os
import sys
import re
import pandas as pd

def infer_differencing(csv_filename):
    match_abs = re.match(r"(\d+)_", csv_filename)  # Matches files starting with a number
    match_rel = re.match(r"Rel(?:_|)(\d+)", csv_filename)  # Matches 'Rel' followed by a number or '_'
    
    if match_abs:
        return f"ABS, {match_abs.group(1)}"
    elif match_rel:
        return f"Rel, {match_rel.group(1)}"
    else:
        return "Default"

def find_top_scoring_models(folder_path):
    # Ensure the provided path is valid
    if not os.path.isdir(folder_path):
        print(f"Error: The provided path '{folder_path}' is not a valid directory.")
        return
    
    # Find all CSV files ending with 'SeasonGrouped.csv'
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('SeasonGrouped.csv')]
    if len(csv_files) != 5:
        print(f"Warning: Expected 5 CSV files, but found {len(csv_files)}.")
    
    top_rows = []
    
    # Iterate through each CSV file and find the row with the lowest Score_SMAPE
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        
        if 'Score_SMAPE' not in df.columns:
            print(f"Error: 'Score_SMAPE' column not found in {csv_file}. Skipping.")
            continue
        
        # Find the row with the lowest Score_SMAPE
        top_row = df.loc[df['Score_SMAPE'].idxmin()]
        
        # Infer Differencing value
        top_row['Differencing'] = infer_differencing(csv_file)
        
        top_rows.append(top_row)
    
    # Create a DataFrame with the top scoring rows
    if top_rows:
        top_df = pd.DataFrame(top_rows)
        output_path = os.path.join(folder_path, 'TopScoringModels.csv')
        top_df.to_csv(output_path, index=False)
        print(f"Top scoring models saved to {output_path}")
    else:
        print("No valid data found to save.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
    else:
        find_top_scoring_models(sys.argv[1])
