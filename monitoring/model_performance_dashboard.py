import streamlit as st
import json
import os
import matplotlib.pyplot as plt
import pandas as pd

def run_performance_dashboard(metrics_path, feature_img_path):
    """Simple UI for monitoring global performance metrics."""
    st.set_page_config(page_title="Churn Performance Monitoring", layout="wide")
    st.title("🛡️ Churn Model Performance Monitoring")
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        st.header("Global Metrics Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        col2.metric("Precision", f"{metrics['precision']:.2f}")
        col3.metric("Recall", f"{metrics['recall']:.2f}")
        col4.metric("F1-Score", f"{metrics['f1_score']:.2f}")
        col5.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")
        
    else:
        st.error(f"Metrics file not found at {metrics_path}. Please run src/evaluate_model.py.")
    
    st.divider()
    
    if os.path.exists(feature_img_path):
        st.header("Top Feature Importances")
        st.image(feature_img_path, caption="Snapshot of Top Predicted Features Drivers Across Dataset")
    else:
        st.warning(f"Importance plot not found at {feature_img_path}.")

if __name__ == "__main__":
    run_performance_dashboard("reports/model_metrics.json", "reports/feature_importance.png")
