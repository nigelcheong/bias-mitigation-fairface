import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def get_paths(project_root):
    """Construct all required paths."""
    return {
        'engineered_csv': project_root / 'data' / 'fairface_features_engineered.csv',
    }

def load_engineered_features(csv_path):
    """Load the engineered features CSV."""
    df = pd.read_csv(csv_path)
    return df

def logistic_regression_model(X, y, feature_names):
    """Train and evaluate logistic regression model."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)

    # Print results
    print("\n" + "=" * 60)
    print("Logistic Regression Results")
    print("=" * 60)
    print(f"\nFeatures used: {feature_names}")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"\nModel accuracy: {model.score(X_test_scaled, y_test):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    # Feature importance
    print(f"\nFeature Coefficients:")
    for name, coef in zip(feature_names, model.coef_[0]):
        print(f"  {name}: {coef:.4f}")

def convert_age_to_numeric(age_str):
    """
    Convert age strings to numeric values.
    Handles formats: '20-30', '10-20', 'more than 70', etc.
    
    Parameters:
    age_str (str): Age string from FairFace dataset
    
    Returns:
    int: Numeric age (uses lower bound for ranges, 75 for 'more than 70')
    """
    if pd.isna(age_str):
        return None
    
    age_str = str(age_str).strip().lower()
    
    # Handle 'more than 70' or 'older'
    if 'more' in age_str or 'older' in age_str:
        return 75  # Use midpoint of 70+
    
    # Handle ranges like '20-30'
    if '-' in age_str:
        try:
            lower = int(age_str.split('-')[0].strip())
            return lower  # Use lower bound
        except ValueError:
            return None
    
    # Handle plain numbers
    try:
        return int(age_str)
    except ValueError:
        return None

def main():
    """Main modeling pipeline."""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    paths = get_paths(project_root)

    print("=" * 60)
    print("Starting Modeling Pipeline")
    print("=" * 60)

    # Load engineered features
    print(f"\n[1/5] Loading engineered features from {paths['engineered_csv']}...")
    df = load_engineered_features(paths['engineered_csv'])
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")

    # Define features and target
    feature_columns = ['brightness', 'contrast', 'sharpness', 'saturation', 'hue_variance']
    
    # Check for missing features
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        print(f"⚠ Missing features: {missing_features}")
        print("Run feature_engineering.py first")
        return

    print(f"\n[2/5] Inspecting age values...")
    print(f"Unique age values:\n{df['age'].unique()}")
    print(f"\nAge value counts:\n{df['age'].value_counts()}")

    print(f"\n[3/5] Converting data types...")
    # Convert age to numeric
    df['age_numeric'] = df['age'].apply(convert_age_to_numeric)
    print(f"Age conversion successful")
    print(f"Age_numeric range: {df['age_numeric'].min()} - {df['age_numeric'].max()}")
    print(f"Missing age values: {df['age_numeric'].isnull().sum()}")

    print(f"\n[4/5] Preparing data...")
    # Check for missing values
    print(f"Missing values before cleaning:\n{df[feature_columns].isnull().sum()}")
    
    # Drop rows with missing feature values or age
    df_clean = df.dropna(subset=feature_columns + ['age_numeric'])
    print(f"Samples after removing missing values: {len(df_clean)}")

    # Create binary target (age >= 30)
    print(f"\nTarget: Age >= 30")
    df_clean['target'] = (df_clean['age_numeric'] >= 30).astype(int)
    target_dist = df_clean['target'].value_counts()
    print(f"Class distribution:\n{target_dist}")
    if len(target_dist) > 1:
        print(f"Ratio (class 1 / class 0): {target_dist[1]/target_dist[0]:.2f}")

    # Sample 10% while maintaining stratification
    print(f"\n[5/5] Sampling 10% of data with stratification...")
    # Use train_test_split to sample 10% with stratification
    df_sample, _ = train_test_split(
        df_clean,
        train_size=0.1,
        random_state=42,
        stratify=df_clean['target']
    )
    print(f"Sample size: {len(df_sample)}")
    print(f"Sample class distribution:\n{df_sample['target'].value_counts()}")

    # Train model
    X = df_sample[feature_columns].values
    y = df_sample['target'].values
    logistic_regression_model(X, y, feature_columns)
    
    print("\n" + "=" * 60)
    print("Modeling complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
