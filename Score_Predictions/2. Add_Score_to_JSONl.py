import os
import json
import numpy as np
from tqdm import tqdm
import sys

# Example function that generates a value based on certain arguments
def evaluate_predictions(predictions, actuals):
    # Convert lists to numpy arrays for easier mathematical operations
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # MAE: Mean Absolute Error
    mae = np.mean(np.abs(predictions - actuals))
    
    # MSE: Mean Squared Error
    mse = np.mean((predictions - actuals) ** 2)
    
    # RMSE: Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # MAPE: Mean Absolute Percentage Error
    # Avoid division by zero by adding a small epsilon to actuals
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.abs((predictions - actuals) / np.where(actuals == 0, np.finfo(float).eps, actuals)) * 100
        mape = np.mean(mape[~np.isnan(mape)])  # Ignore NaN values caused by 0/0 divisions
    
    # R-squared (Coefficient of Determination)
    ss_total = np.sum((actuals - np.mean(actuals)) ** 2)
    ss_residual = np.sum((predictions - actuals) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    
    # GFD: This is typically a custom metric; here we calculate a general deviation metric
    global_deviance = np.sum(np.abs(predictions - actuals)) / len(actuals)
    
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R-squared": r_squared,
        "Global Forecast Deviance": global_deviance
    }

def update_jsonl_with_placeholder(folder_path):
    # Iterate over each file in the input folder
    for filename in tqdm(os.listdir(folder_path), desc="Processing JSON files"):
        file_path = os.path.join(folder_path, filename)
        
        # Only process .jsonl files
        if filename.endswith(".jsonl"):
            updated_records = []  # List to store records for each file
            updated = False  # Flag to check if any record was updated
            
            # Read and process each JSON object (line) in the file
            with open(file_path, "r") as infile:
                for line in infile:
                    record = json.loads(line)  # Parse JSON string to dictionary
                    
                    # Check if "Score" is present
                    if "Score" not in record:
                        # Obtain required arguments for the function from the record
                        Forecast = record.get("forecast_values")  # Replace with the actual key names
                        Target = record.get("actual_values")
                        
                        if Forecast is not None and Target is not None:
                            # Adjust lengths if necessary
                            if len(Forecast) > len(Target):
                                Forecast = Forecast[1:]  # Remove the first value of Forecast
                            elif len(Forecast) < len(Target):
                                # Handle cases where Forecast is shorter than Target
                                record["Score"] = {"error": "Forecast length is shorter than Target"}
                                updated = True
                                updated_records.append(record)
                                continue
                            
                            # Calculate the score
                            record["Score"] = evaluate_predictions(Forecast, Target)
                            updated = True  # Set flag to True since we updated this record
                        
                        else:
                            record["Score"] = {"error": "Missing forecast_values or actual_values"}
                            updated = True
                    
                    # Append the record (updated or not) to the list
                    updated_records.append(record)
            
            # Write the updated records back to the same file only if updates were made
            if updated:
                with open(file_path, "w") as outfile:
                    for record in updated_records:
                        outfile.write(json.dumps(record) + "\n")

# Usage
# Load all the files in the output folder
if len(sys.argv) != 2:
    print("Usage: python3 2. Add_Score_to_jsonl folder_path")
    sys.exit(1)

# Get paths from command-line arguments
folder_path = sys.argv[1]
update_jsonl_with_placeholder(folder_path)
