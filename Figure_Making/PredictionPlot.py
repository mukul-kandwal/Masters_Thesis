import sys
import csv
import ast
from datetime import timedelta
from dateutil.parser import parse as parse_date
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter

# Configuration Section - Modify these values as needed
CONFIG = {
    "y_label": "Flu Cases (x1000)",
    "x_label": "",
    "line_color": "#1f77b4",
    "background_color": "white",
    "forecast_color": "#d62728",
    "figure_size": (10, 6),
    "font_size": 12,
    "line_width": 3,
    "dot_size": 25,
    "line-dot-size" : 7,
    "grid_visible": False,
    # "grid_style": "--",
    "title": "Actual Flu Case Number for 24-25 Season",

}

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_csv> <start_date>")
        sys.exit(1)

    input_csv = sys.argv[1]
    start_date_str = sys.argv[2]

    # Read and parse CSV data
    try:
        with open(input_csv, 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader)
            actual_values = ast.literal_eval(row['actual_values'])
            forecast_values = ast.literal_eval(row['forecast_values'])
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        sys.exit(1)

    if len(actual_values) != len(forecast_values):
        print("Error: actual_values and forecast_values must be the same length")
        sys.exit(1)

    # Generate dates and labels
    try:
        start_date = parse_date(start_date_str)
    except Exception as e:
        print(f"Error parsing start date: {e}")
        sys.exit(1)

    dates = [start_date + timedelta(days=7*i) for i in range(len(actual_values))]
    
    # Create x-axis labels: show the full month name only when the month changes; otherwise, leave blank
    x_labels = []
    for i, date in enumerate(dates):
        if i == 0 or date.month != dates[i-1].month:
            label = date.strftime('%b')
        else:
            label = ""
        x_labels.append(label)

    # Create plot
    plt.figure(figsize=CONFIG['figure_size'])
    plt.rcParams['font.size'] = CONFIG['font_size']
    ax = plt.gca()
    ax.set_facecolor(CONFIG['background_color'])

    # Plot actual values as line with dot markers for clarity
    ax.plot(dates, actual_values, 
            color=CONFIG['line_color'], 
            linewidth=CONFIG['line_width'], 
            marker='o',
            markersize=CONFIG['line-dot-size'],
            label='Actual Values')

    # Plot forecast values as dots (skip first)
    # if len(forecast_values) > 1:
    #     ax.scatter(dates[1:], forecast_values[1:], 
    #                color=CONFIG['forecast_color'], 
    #                s=CONFIG['dot_size'], 
    #                zorder=5, 
    #                label='Forecast Values')

    # Configure plot
    ax.set_xlabel(CONFIG['x_label'])
    ax.set_ylabel(CONFIG['y_label'])
    ax.set_title(CONFIG['title'])
    ax.set_xticks(dates)
    ax.set_xticklabels(x_labels, ha='center')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2)
    ax.grid(axis='y', alpha=0.2)
    if CONFIG['grid_visible']:
        ax.grid(True, linestyle=CONFIG['grid_style'])

    # Format y-axis to show values in thousands
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1000:g}'))

    plt.tight_layout()

    # Save to PNG
    output_path = os.path.splitext(input_csv)[0] + '.png'
    plt.savefig(output_path)
    print(f"Graph saved to: {output_path}")

if __name__ == "__main__":
    main()
