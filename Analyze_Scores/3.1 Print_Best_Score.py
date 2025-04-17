import sys
import os
import json
import re

def extract_and_print_first_dicts(folder_path):
    # Summary dictionary to collect score values
    summary = {}

    # Loop through all files in the folder that match the prefix
    for filename in os.listdir(folder_path):
        if filename.startswith("best_scores_Score_") and filename.endswith(".txt"):
            # Extract the score type from the filename
            score_type = filename.split("best_scores_Score_")[-1].replace(".txt", "")
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Extract content starting from line 3
            json_like_content = ''.join(lines[2:]).strip()

            # Clean up JSON-like content
            json_like_content = json_like_content.replace("'", '"')
            json_like_content = json_like_content.replace("None", "null")
            json_like_content = json_like_content.replace("True", "true").replace("False", "false")

            # Parse JSON
            try:
                data_list = json.loads(json_like_content)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON data in {filename}: {e}")
                continue

            # Check if data_list is valid
            if not isinstance(data_list, list) or not data_list:
                print(f"The JSON data in {filename} is not a list or is empty.")
                continue

            # Extract the first dictionary
            first_dict = data_list[0]

            # Get the exact matching score value for the summary
            score_key = f"Score_{score_type}"
            if score_key in first_dict:
                summary[score_key] = first_dict[score_key]

            # Print header for the file
            print("\n" + "=" * 40)
            print(f"File: {filename}")
            print("=" * 40)

            # Print each key-value pair if it meets the condition
            for key, value in first_dict.items():
                if not key.startswith("Score_") or key == score_key:
                    print(f"{key}: {value}")

    # Print summary section at the top, sorted by score type
    print("\nSummary of Scores:")
    print("=" * 40)
    for score, value in sorted(summary.items()):
        print(f"{score}: {value}")

if __name__ == "__main__":
    # Pass folder path as a system argument
    folder_path = sys.argv[1]
    extract_and_print_first_dicts(folder_path)