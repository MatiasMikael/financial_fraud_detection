import os
import kaggle

# Define dataset and output path
dataset_name = "sriharshaeedala/financial-fraud-detection-dataset"
output_path = "2_raw_data"

# Create output folder if not exists
os.makedirs(output_path, exist_ok=True)

# Download dataset
kaggle.api.dataset_download_files(dataset_name, path=output_path, unzip=True)

print(f"Dataset downloaded to {output_path}")