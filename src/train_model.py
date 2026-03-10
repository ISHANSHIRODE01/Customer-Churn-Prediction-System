import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from data_preprocessing import load_data, clean_data, optimize_memory
from feature_engineering import build_preprocessing_pipeline

def train_churn_model(raw_data_path, model_output_path):
    """Full end-to-end training pipeline."""
    # 1. Load and Preprocess Data
    df = load_data(raw_data_path)
    df = clean_data(df)
    df = optimize_memory(df)
    
    # Target Handling
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Build Architecture
    preprocessor = build_preprocessing_pipeline(X_train)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    # Final Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # 4. Train Model
    pipeline.fit(X_train, y_train)
    
    # 5. Save Artifacts
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"Model successfully trained and saved to {model_output_path}")

if __name__ == "__main__":
    train_churn_model("data/raw/telco_customer_churn.csv", "models/churn_model.pkl")
