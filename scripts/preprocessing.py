
import zipfile
import os
import pandas as pd
import numpy as np

from pathlib import Path
from PIL import Image

# Zip files
zip_025 = '../data/fairface-img-margin025-trainval.zip'
zip_125 = '../data/fairface-img-margin125-trainval.zip'

# Extraction directories
extract_dir_025 = '../data/fairface_025'
extract_dir_125 = '../data/fairface_125'

def extract_zip(zip_path, extract_to):
    """
    Extract a zip file to a specified directory if the directory does not already exist.
    
    Parameters:
    zip_path (str or Path): The path to the zip file.
    extract_to (str or Path): The directory to extract the contents to.
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    
    if not extract_to.exists():
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted to {extract_to}")
    else:
        print(f"{extract_to} already exists, skipping extraction.")

# Extract both datasets
extract_zip(zip_025, extract_dir_025)
extract_zip(zip_125, extract_dir_125)

# Load train and validation labels
train_labels = pd.read_csv('../data/fairface_label_train.csv')
val_labels = pd.read_csv('../data/fairface_label_val.csv')

# Check for missing values
print(df.isnull().sum())

# Combine for overall statistics
df = pd.concat([train_labels, val_labels], ignore_index=True)
print(df.shape)
df.head()

# Export the combined dataframe to a CSV file
df.to_csv('../data/fairface_combined_labels.csv', index=False)
