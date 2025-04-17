#!/usr/bin/env python3
import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 9,
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
})

def render_styled_table(data, output_path):
    """Create styled table with dynamic row heights"""
    # Calculate row heights
    row_heights = [0.4]  # Header height
    for param_str in data['Hyperparameters']:
        lines = len(param_str.split('\n')) + 1
        row_heights.append(0.2 * lines)

    fig_height = sum(row_heights)
    fig, ax = plt.subplots(figsize=(7, fig_height))
    ax.axis('off')

    # Create table content
    cell_text = []
    for _, row in data.iterrows():
        formatted_params = []
        for line in row['Hyperparameters'].split('\n'):
            if ': ' in line:
                param, value = line.split(': ', 1)
                # Convert PascalCase to spaced words and add LaTeX spaces
                param_spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', param)
                param_spaced = param_spaced.replace(' ', r'\ ')  # Add LaTeX spaces
                formatted_params.append(r'$\mathbf{{{param}}}$: {value}'.format(param=param_spaced, value=value))
            else:
                formatted_params.append(line)
        cell_text.append([row['Model'], '\n'.join(formatted_params)])

    # Create basic table
    table = ax.table(
        cellText=cell_text,
        colLabels=['Model', 'Hyperparameters'],
        loc='center',
        cellLoc='left',
        colWidths=[0.2, 0.8]
    )

    # Adjust row heights and vertical alignment
    for row_idx in range(len(row_heights)):
        for col_idx in range(2):
            cell = table[row_idx, col_idx]
            cell.set_height(row_heights[row_idx]/fig_height)
            
            # Set vertical alignment for model column
            if col_idx == 0 and row_idx > 0:
                cell.get_text().set_verticalalignment('top')
                cell.get_text().set_linespacing(1.5)

    # Style adjustments
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Header styling
    for col_idx in range(2):
        cell = table[0, col_idx]
        cell.set_facecolor('#4F81BD')
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')

    # Body styling
    for row_idx in range(1, len(row_heights)):
        for col_idx in range(2):
            cell = table[row_idx, col_idx]
            cell.set_facecolor('#FFFFFF' if row_idx%2 == 1 else '#F1F1F1')
            cell.set_edgecolor('#E0E0E0')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_excel_file(input_path, output_path):
    try:
        df = pd.read_excel(input_path, sheet_name='Sheet1')
        df = df[['Model', 'Hyperparameters']]
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    formatted_data = []
    for _, row in df.iterrows():
        params = []
        for line in str(row['Hyperparameters']).split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
            if '. ' in line:
                line = line.split('. ', 1)[1]
            params.append(line)
        
        formatted_data.append({
            'Model': row['Model'],
            'Hyperparameters': '\n'.join(params)
        })

    render_styled_table(pd.DataFrame(formatted_data), output_path)

def main():
    parser = argparse.ArgumentParser(description='Create tables from Excel')
    parser.add_argument('input', help='Input Excel file')
    parser.add_argument('output', help='Output PNG file')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        sys.exit(1)

    process_excel_file(args.input, args.output)
    print(f"Created: {args.output}")

if __name__ == "__main__":
    main()