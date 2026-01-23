import zipfile
import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

def get_paths(project_root):
    """Construct and return all required paths."""
    return {
        'zip_025': project_root / 'data' / 'fairface-img-margin025-trainval.zip',
        'zip_125': project_root / 'data' / 'fairface-img-margin125-trainval.zip',
        'extract_dir_025': project_root / 'data' / 'fairface_025',
        'extract_dir_125': project_root / 'data' / 'fairface_125',
        'train_labels_csv': project_root / 'data' / 'fairface_label_train.csv',
        'val_labels_csv': project_root / 'data' / 'fairface_label_val.csv',
        'output_csv': project_root / 'data' / 'fairface_combined_labels.csv',
    }

def extract_zip(zip_path, extract_to):
    """
    Extract a zip file to a specified directory if it doesn't already exist.
    
    Parameters:
    zip_path (Path or str): The path to the zip file.
    extract_to (Path or str): The directory to extract the contents to.
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

def load_and_combine_labels(train_csv, val_csv):
    """Load and combine train/validation label CSVs."""
    train_labels = pd.read_csv(train_csv)
    val_labels = pd.read_csv(val_csv)
    df = pd.concat([train_labels, val_labels], ignore_index=True)
    return df

def main():
    """Main preprocessing pipeline."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent  # Go up one level to project root
    
    # Get all paths
    paths = get_paths(project_root)
    
    print("=" * 60)
    print("Starting FairFace Preprocessing Pipeline")
    print("=" * 60)
    
    # Extract datasets
    print("\n[1/3] Extracting datasets...")
    extract_zip(paths['zip_025'], paths['extract_dir_025'])
    extract_zip(paths['zip_125'], paths['extract_dir_125'])
    
    # Load and combine labels
    print("\n[2/3] Loading and combining labels...")
    df = load_and_combine_labels(paths['train_labels_csv'], paths['val_labels_csv'])
    print(f"Combined dataset shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    
    # Check data quality
    print("\n[3/3] Checking data quality...")
    missing = df.isnull().sum()
    print(f"Missing values:\n{missing[missing > 0] if missing.any() else 'None'}")
    
    # Save combined labels
    df.to_csv(paths['output_csv'], index=False)
    print(f"\n✓ Saved combined labels to {paths['output_csv']}")
    print("=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
    