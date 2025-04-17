#!/usr/bin/env python3
import sys
import os
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=12,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], ax=None):
    """
    Create a publication-ready table using matplotlib.
    Adapted from: https://blog.martisak.se/2021/04/10/publication_ready_tables/
    """
    if ax is None:
        # Compute the figure size from the DataFrame shape
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns,
                         cellLoc='center', loc='center')
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for key, cell in mpl_table.get_celld().items():
        cell.set_edgecolor(edge_color)
        # Header row styling
        if key[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[key[0] % len(row_colors)])
    return ax.get_figure(), ax

def process_csv_file(csvfile, rows):
    print(f"\nProcessing file: {csvfile}")
    try:
        df = pd.read_csv(csvfile)
    except Exception as e:
        print(f"Error reading CSV file {csvfile}: {e}")
        return

    # Automatically drop columns with only 1 unique value (ignoring NaNs)
    single_val_cols = [col for col in df.columns if df[col].dropna().nunique() == 1]
    if single_val_cols:
        print(f"Automatically dropping columns with only one unique value: {single_val_cols}")
        df.drop(columns=single_val_cols, inplace=True)

    # Process remaining columns: for Score_ columns, only keep Score_SMAPE
    cols_to_drop = []
    for col in df.columns:
        if col.startswith("Score_"):
            if col == "Score_SMAPE":
                continue  # Keep Score_SMAPE
            else:
                cols_to_drop.append(col)
                continue
    df.drop(columns=cols_to_drop, inplace=True)

    # Format Score_SMAPE column to 2 decimal places if present.
    if "Score_SMAPE" in df.columns:
        df["Score_SMAPE"] = df["Score_SMAPE"].round(2)

    # Sorting: sort the DataFrame in ascending order using Score_MAPE if available,
    # otherwise try Score_SMAPE.
    sort_col = None
    if "Score_MAPE" in df.columns:
        sort_col = "Score_MAPE"
    elif "Score_SMAPE" in df.columns:
        sort_col = "Score_SMAPE"
    if sort_col:
        try:
            df.sort_values(by=sort_col, ascending=True, inplace=True)
        except Exception as e:
            print(f"Error sorting by column {sort_col}: {e}")
    else:
        print("No Score_MAPE or Score_SMAPE column found; skipping sort.")

    # Prepare DataFrame for display based on the --rows option.
    display_df = df.copy()
    extra_row_text_template = "... ({} More Rows)"
    if rows is not None and len(df) > rows:
        display_df = df.head(rows).copy()
        remaining = len(df) - rows
        extra_row = {col: "" for col in display_df.columns}
        first_col = list(display_df.columns)[0]
        extra_row[first_col] = extra_row_text_template.format(remaining)
        display_df = pd.concat([display_df, pd.DataFrame([extra_row])], ignore_index=True)

    # Remove prefix from column headers (everything before first underscore)
    new_columns = {}
    for col in display_df.columns:
        new_columns[col] = col.split("_", 1)[-1] if "_" in col else col
    display_df.rename(columns=new_columns, inplace=True)

    # Generate LaTeX table using pandas' to_latex() function.
    latex_table = display_df.to_latex(index=False, escape=False)
    print("\nGenerated LaTeX Table:")
    print(latex_table)

    # Create publication-ready table using the custom function.
    fig, ax = render_mpl_table(display_df, col_width=3.0, row_height=0.625, font_size=12)

    # Save the figure as an image file in the same path as the CSV.
    base, _ = os.path.splitext(csvfile)
    image_file = base + ".png"
    fig.savefig(image_file, bbox_inches='tight', dpi=300)
    print(f"\nTable image saved to: {image_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate publication ready tables from CSV files ending with SeasonGrouped.csv in a folder.'
    )
    parser.add_argument('folder', help='Path to the folder containing CSV files ending with SeasonGrouped.csv')
    parser.add_argument('--rows', type=int, default=None, help='Number of rows to display in the table.')
    args = parser.parse_args()

    folder_path = args.folder
    pattern = os.path.join(folder_path, '*SeasonGrouped.csv')
    csv_files = glob.glob(pattern)
    if not csv_files:
        print(f"No CSV files ending with 'SeasonGrouped.csv' found in folder {folder_path}")
        sys.exit(1)

    for csvfile in csv_files:
        process_csv_file(csvfile, args.rows)

if __name__ == "__main__":
    main()
