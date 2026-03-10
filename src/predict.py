import joblib
import pandas as pd
import numpy as np

def make_predictions(model_path, input_df):
    """Loads a model and makes predictions on an input dataframe."""
    pipeline = joblib.load(model_path)
    
    # Ensure TotalCharges is numeric as expected by the pipeline's internal logic
    if 'TotalCharges' in input_df.columns:
        input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce').fillna(0)
    
    predictions = pipeline.predict(input_df)
    probabilities = pipeline.predict_proba(input_df)[:, 1]
    
    return predictions, probabilities

if __name__ == "__main__":
    # Test on single instance
    test_data = {
        'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'tenure': 24, 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': 'One year', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check', 'MonthlyCharges': 60.0, 'TotalCharges': 1500.0
    }
    input_df = pd.DataFrame([test_data])
    preds, probs = make_predictions("models/churn_model.pkl", input_df)
    print(f"Prediction: {preds[0]}, Churn Probability: {probs[0]:.2%}")
