#!/usr/bin/env python3
import sys
import os
import pandas as pd
import ast
import logging
from pprint import pprint
import numpy as np

def process_file(file_path):
    logging.info(f"\n=== Processing CSV file: {file_path} ===")
    try:
        df = pd.read_csv(file_path)
        df['Score_SMAPE_Residuals'] = df['Score_SMAPE_Residuals'].apply(ast.literal_eval)
    except Exception as e:
        logging.error(f"Failed to read CSV file {file_path}: {e}")
        return

    logging.info("CSV columns found: " + ", ".join(df.columns))
    
    # Identify metadata columns (those that start with "metadata_"), excluding 'metadata_Training_Restriction'
    metadata_columns = [col for col in df.columns if col.startswith("metadata_") and col != "metadata_Training_Restriction"]
    logging.info("Found metadata columns: " + ", ".join(metadata_columns))
    
    # Determine grouping keys: all metadata columns except metadata_Season.
    grouping_keys = [col for col in metadata_columns if col != "metadata_Season"]
    logging.info("Grouping by keys (all metadata columns except 'metadata_Season'): " + ", ".join(grouping_keys))
    
    # Check that our special column exists.
    if "Score_SMAPE_Residuals" not in df.columns:
        logging.error(f"Column 'Score_SMAPE_Residuals' not found in CSV {file_path}. Skipping this file.")
        return

    # List which columns will be kept.
    used_columns = grouping_keys + ["Score_SMAPE_Residuals"]
    dropped_columns = [col for col in df.columns if col not in used_columns]
    logging.info("Dropping non-essential columns: " + ", ".join(dropped_columns))

    # Create a subset DataFrame containing only the relevant columns.
    df_subset = df[used_columns].copy()
    pprint(df_subset)
    
    # Convert any list-type values in grouping keys to tuples so they are hashable.
    for col in grouping_keys:
        if df_subset[col].apply(lambda x: isinstance(x, list)).any():
            df_subset[col] = df_subset[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    # Group the rows by the grouping keys.
    grouped = df_subset.groupby(grouping_keys).agg(Score_SMAPE_Residuals=('Score_SMAPE_Residuals', 'sum'), Score_SMAPE=('Score_SMAPE_Residuals', lambda x: np.mean(np.concatenate(x.tolist())))).reset_index()
    pprint(grouped)
    logging.info(f"Found {len(grouped)} unique grouping configurations based on metadata (excluding metadata_Season).")
    
    # Optionally, convert the merged list into a string so it is saved nicely in the CSV.
    out_df = grouped
    out_df["Score_SMAPE_Residuals"] = out_df["Score_SMAPE_Residuals"].apply(lambda x: str(x))

    # Create the output file name by appending _SeasonGrouped before the extension.
    base, ext = os.path.splitext(file_path)
    output_file = f"{base}_SeasonGrouped.csv"
    try:
        out_df.to_csv(output_file, index=False)
        logging.info(f"Output CSV saved to: {output_file}")
    except Exception as e:
        logging.error(f"Failed to write output CSV for {file_path}: {e}")

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Check for input argument.
    if len(sys.argv) < 2:
        logging.error("Usage: python group_csv.py <input_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    if not os.path.isdir(input_folder):
        logging.error(f"The provided path '{input_folder}' is not a valid directory.")
        sys.exit(1)

    # List CSV files in the folder that do not end with "SeasonGrouped.csv"
    files = os.listdir(input_folder)
    csv_files = [
        filename for filename in files
        if filename.endswith('.csv') and not filename.endswith('SeasonGrouped.csv')
    ]

    if not csv_files:
        logging.info("No applicable CSV files found in the folder.")
        sys.exit(0)

    # Process each CSV file.
    for filename in csv_files:
        file_path = os.path.join(input_folder, filename)
        process_file(file_path)

if __name__ == '__main__':
    main()
