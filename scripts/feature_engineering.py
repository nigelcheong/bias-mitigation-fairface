import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def get_paths(project_root):
    """Construct all required paths."""
    return {
        'input_csv': project_root / 'data' / 'fairface_combined_labels.csv',
        'image_dir': project_root / 'data' / 'fairface_025',
        'output_csv': project_root / 'data' / 'fairface_features_engineered.csv',
    }

def brightness(img_path):
    """Calculate mean brightness."""
    try:
        img = Image.open(img_path).convert('L')
        return float(np.array(img).mean())
    except Exception:
        return None

def contrast(img_path):
    """Calculate contrast (std dev of pixel intensities)."""
    try:
        img = Image.open(img_path).convert('L')
        return float(np.array(img).std())
    except Exception:
        return None

def sharpness(img_path):
    """Calculate sharpness (Laplacian variance)."""
    try:
        img = Image.open(img_path).convert('L')
        image_array = np.array(img)
        laplacian = (np.abs(np.gradient(np.gradient(image_array)[0])[0]) + 
                     np.abs(np.gradient(np.gradient(image_array)[1])[1]))
        return float(laplacian.var())
    except Exception:
        return None

def saturation(img_path):
    """Calculate mean saturation in HSV space."""
    try:
        img = Image.open(img_path).convert('RGB')
        img_hsv = img.convert('HSV')
        saturation_channel = np.array(img_hsv)[:, :, 1]
        return float(saturation_channel.mean())
    except Exception:
        return None

def hue_variance(img_path):
    """Calculate hue channel variance."""
    try:
        img = Image.open(img_path).convert('RGB')
        img_hsv = img.convert('HSV')
        hue_channel = np.array(img_hsv)[:, :, 0]
        return float(hue_channel.var())
    except Exception:
        return None

def main():
    """Main feature engineering pipeline."""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    paths = get_paths(project_root)
    
    print("=" * 60)
    print("Starting Feature Engineering Pipeline")
    print("=" * 60)
    
    # Load data
    print(f"\n[1/6] Loading {paths['input_csv']}...")
    df = pd.read_csv(paths['input_csv'])
    print(f"Loaded {len(df)} rows")
    
    # Compute features
    tqdm.pandas(desc="Computing features")
    
    print("\n[2/6] Computing brightness...")
    df['brightness'] = df['file'].progress_apply(
        lambda f: brightness(paths['image_dir'] / f)
    )
    
    print("[3/6] Computing contrast...")
    df['contrast'] = df['file'].progress_apply(
        lambda f: contrast(paths['image_dir'] / f)
    )
    
    print("[4/6] Computing sharpness...")
    df['sharpness'] = df['file'].progress_apply(
        lambda f: sharpness(paths['image_dir'] / f)
    )
    
    print("[5/6] Computing saturation...")
    df['saturation'] = df['file'].progress_apply(
        lambda f: saturation(paths['image_dir'] / f)
    )
    
    print("[6/6] Computing hue variance...")
    df['hue_variance'] = df['file'].progress_apply(
        lambda f: hue_variance(paths['image_dir'] / f)
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Feature Summary:")
    print(df[['brightness', 'contrast', 'sharpness', 'saturation', 'hue_variance']].describe())
    print(f"\nMissing values:\n{df[['brightness', 'contrast', 'sharpness', 'saturation', 'hue_variance']].isnull().sum()}")
    
    # Save
    df.to_csv(paths['output_csv'], index=False)
    print(f"\n✓ Saved {len(df)} rows with 6 features to {paths['output_csv']}")
    print("=" * 60)

if __name__ == '__main__':
    main()
