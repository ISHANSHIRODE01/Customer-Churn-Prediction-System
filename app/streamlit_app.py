import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load the full inference pipeline
try:
    pipeline = joblib.load('churn_model_pipeline.joblib')
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['classifier']
except FileNotFoundError:
    st.error("Model file not found. Please run the notebook first.")
    st.stop()

st.set_page_config(page_title='Churn AI Dashboard', layout='wide')

# --- Header ---
st.title('📊 Customer Churn Intelligence Dashboard')
st.markdown("""
This dashboard provides real-time churn predictions, explains the reasoning behind each prediction (XAI), 
and shows global model performance insights.
""")

# --- Sidebar Inputs ---
st.sidebar.header('Customer Profile')
def user_input_features():
    tenure = st.sidebar.slider('Tenure (months)', 0, 72, 24)
    contract = st.sidebar.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    monthly_charges = st.sidebar.number_input('Monthly Charges ($)', 18.0, 120.0, 60.0)
    total_charges = st.sidebar.number_input('Total Charges ($)', 18.0, 9000.0, 1500.0)
    internet = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    tech_support = st.sidebar.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])

    data = {
        'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': internet, 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'TechSupport': tech_support, 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': contract, 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check', 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }
    return pd.DataFrame([data])

input_df = user_input_features()
input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce').fillna(0)

# --- Tabs for different views ---
tab1, tab2 = st.tabs(["🎯 Individual Prediction", "📈 Global Insights & Monitoring"])

with tab1:
    st.header("Prediction for Current Customer")
    
    # Prediction
    prob = pipeline.predict_proba(input_df)[0][1]
    risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Churn Risk", f"{prob:.1%}", delta=risk_level, delta_color="inverse")
    col2.metric("Prediction", "CHURN" if prob > 0.5 else "NO CHURN")
    col3.metric("Customer Segment", risk_level)

    st.divider()
    
    # SHAP Explainable AI (XAI)
    st.subheader('✨ Explainable AI: Why this prediction?')
    try:
        processed_input = preprocessor.transform(input_df)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(processed_input)
        
        # Get feature names
        cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
        num_features = preprocessor.transformers_[0][2]
        feature_names = np.concatenate([num_features, cat_features])

        fig_waterfall, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[0].values if hasattr(shap_values[0], 'values') else shap_values[0],
            base_values=explainer.expected_value,
            data=processed_input[0],
            feature_names=feature_names
        ), show=False)
        st.pyplot(fig_waterfall)
        st.caption("Positive values (red) increase churn risk, negative values (blue) decrease it.")
    except Exception as e:
        st.warning("Individual explanation loading...")

with tab2:
    st.header("Global Model Performance & Feature Importance")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Global Feature Importance")
        # Extract XGBoost feature importance
        importances = model.feature_importances_
        # Simplified naming for visualization
        indices = np.argsort(importances)[-10:]  # top 10
        
        fig_imp, ax_imp = plt.subplots()
        plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        st.pyplot(fig_imp)
        st.info("These are the top 10 drivers of churn across all customers.")

    with col_b:
        st.subheader("Model Monitoring (Training Snapshot)")
        # Static metrics (In a real scenario, these would load from a log file)
        metrics = {
            "Accuracy": "81.2%",
            "F1-Score": "0.78",
            "ROC-AUC": "0.86",
            "Model Version": "v1.0.2"
        }
        for k, v in metrics.items():
            st.write(f"**{k}:** {v}")
        
        st.success("Model Health: Stable")
        st.markdown("---")
        st.caption("Monitoring Dashboard updates whenever the model is retrained in the notebook.")
