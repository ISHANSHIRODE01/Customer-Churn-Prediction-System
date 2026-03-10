import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """Loads raw CSV data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    return pd.read_csv(file_path)

def optimize_memory(df):
    """Optimizes memory by downcasting numerical columns."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if str(df[col].dtype).startswith('int'):
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    return df

def clean_data(df):
    """Handles missing values and specific Telco churn cleaning steps."""
    # Convert TotalCharges to numeric (it has some empty strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Impute missing TotalCharges with median
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    # Drop customerID as it's not a predictor
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    return df

if __name__ == "__main__":
    # Example execution
    raw_path = "data/raw/telco_customer_churn.csv"
    processed_path = "data/processed/cleaned_data.csv"
    
    df = load_data(raw_path)
    df = clean_data(df)
    df = optimize_memory(df)
    
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Data cleaned and saved to {processed_path}")
