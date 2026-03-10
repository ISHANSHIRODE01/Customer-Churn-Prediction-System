import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_preprocessing import load_data, clean_data, optimize_memory

def evaluate_performance(model_path, data_path, output_report_path, output_img_path):
    """Evaluates the model on the full original dataset and generates metrics/plots."""
    # 1. Load Data and Process Setup
    df = load_data(data_path)
    df = clean_data(df)
    df = optimize_memory(df)
    
    # Target Handling
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # 2. Load Model Pipeline
    pipeline = joblib.load(model_path)
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]
    
    # 3. Calculate Metrics
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "f1_score": float(f1_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_prob))
    }
    
    # Save Metrics JSON
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # 4. Feature Importance Feature Plot for Reports
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Extract feature names from preprocessing pipeline
    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    num_features = preprocessor.transformers_[0][2]
    all_feature_names = list(num_features) + list(cat_features)
    
    feat_importances = pd.Series(classifier.feature_importances_, index=all_feature_names)
    plt.figure(figsize=(10, 6))
    feat_importances.nlargest(10).plot(kind='barh', title="Top 10 Feature Importances")
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(output_img_path)
    plt.close()
    print(f"Metrics saved to {output_report_path}")
    print(f"Initial Importance Plot saved to {output_img_path}")

if __name__ == "__main__":
    evaluate_performance(
        "models/churn_model.pkl",
        "data/raw/telco_customer_churn.csv",
        "reports/model_metrics.json",
        "reports/feature_importance.png"
    )
