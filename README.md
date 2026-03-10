<div align="center">

# 🔮 Customer Churn Prediction System

### A Production-Grade Machine Learning Pipeline for Telecom Churn Intelligence

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-XAI-brightgreen?style=for-the-badge)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.3-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/Tests-Passing-success?style=for-the-badge&logo=pytest)](https://pytest.org/)

---

*A fully reproducible, modular, and explainable machine learning system that predicts customer churn with high accuracy, serving real-time decisions through a Streamlit dashboard and containerized for one-command cloud deployment.*

[Live Demo](#-streamlit-application-overview) • [Architecture](#-system-architecture) • [Quick Start](#-installation-guide) • [Model Results](#-model-evaluation-metrics) • [Docker Deploy](#-docker-deployment)

</div>

---

## 📌 Project Overview

Customer churn — when a subscriber terminates their service — costs the telecommunications industry billions annually. This project delivers a **production-grade, end-to-end machine learning system** built on the [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), designed to give data-driven teams real-time churn risk scores and interpretable explanations at individual customer level. The system implements **Scikit-Learn Pipelines** for airtight data preprocessing (preventing data leakage), trains and compares three canonical ML classifiers (**Logistic Regression**, **Random Forest**, **XGBoost**), delivers local and global explanations using **SHAP**, and exposes the entire solution through an interactive **Streamlit** dashboard — fully containerized with **Docker** for seamless deployment to any cloud provider.

This project was engineered to demonstrate production ML best practices: **modular code design**, **automated testing**, **model serialization**, **explainability reporting**, and **monitoring dashboards** — meeting the bar for ML Engineering roles at top-tier technology organizations.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔁 **End-to-End Pipeline** | Data ingestion → Preprocessing → Training → Evaluation → Serving |
| 🛡️ **Leakage-Safe Engineering** | All transformations gated within `sklearn.Pipeline` — zero test-set leakage |
| 🧪 **Multi-Model Benchmark** | Logistic Regression, Random Forest, and XGBoost compared on 5 metrics |
| 🧠 **Explainable AI (SHAP)** | Global importance plots and per-customer waterfall explanations |
| 📊 **Live Dashboard** | Streamlit app for real-time churn risk scoring with SHAP visualizations |
| 🛰️ **Model Monitoring** | Dedicated Streamlit dashboard for performance tracking and drift alerting |
| 🐳 **Docker Ready** | One-command container launch for local or cloud deployment |
| ✅ **Automated Testing** | `pytest` test suite for data pipeline integrity validation |
| 📁 **Modular Architecture** | `src/` contains fully importable, testable Python modules |

---

## 🎬 Demo / Example Output

### Real-Time Churn Prediction

```
Customer Profile:
  ├── Tenure:          6 months
  ├── Contract:        Month-to-month
  ├── Internet Service: Fiber optic
  └── Monthly Charges: $85.00

Prediction:    ⚠️  HIGH CHURN RISK
Probability:   74.3%
Risk Tier:     High
```

### SHAP Waterfall Explanation

```
Base churn rate → 0.27
  + Contract: Month-to-month    → +0.18 (increases risk)
  + Tenure: 6 months            → +0.14 (short = high risk)
  + MonthlyCharges: $85         → +0.09 (high bill)
  - TechSupport: Yes            → -0.06 (decreases risk)
  = Final Prediction            → 0.743
```

> **Note**: Screenshots of the live dashboards are available in `reports/`. Run the dashboards locally using the instructions below to explore interactive visuals.

---

## 🏗️ System Architecture

```
                        ┌──────────────────────────────────────────────────┐
                        │              Data Layer                           │
                        │  data/raw/telco_customer_churn.csv               │
                        └───────────────────┬──────────────────────────────┘
                                            │
                                            ▼
                        ┌──────────────────────────────────────────────────┐
                        │            Preprocessing Module                  │
                        │  src/data_preprocessing.py                       │
                        │  ├── load_data()       → Raw CSV ingestion       │
                        │  ├── clean_data()      → Type coercion, nulls    │
                        │  └── optimize_memory() → Dtype downcasting       │
                        └───────────────────┬──────────────────────────────┘
                                            │
                                            ▼
                        ┌──────────────────────────────────────────────────┐
                        │           Feature Engineering Module             │
                        │  src/feature_engineering.py                      │
                        │  ├── Numerical: Median Imputer + StandardScaler  │
                        │  ├── Categorical: Mode Imputer + OneHotEncoder   │
                        │  └── sklearn ColumnTransformer (no leakage)      │
                        └───────────────────┬──────────────────────────────┘
                                            │
                                            ▼
                        ┌──────────────────────────────────────────────────┐
                        │             Model Training Layer                 │
                        │  src/train_model.py                              │
                        │  ├── LogisticRegression  (baseline)              │
                        │  ├── RandomForestClassifier                      │
                        │  └── XGBClassifier (primary / best)             │
                        └───────────────────┬──────────────────────────────┘
                                            │
                               ┌────────────┴───────────┐
                               ▼                        ▼
         ┌─────────────────────────────┐  ┌─────────────────────────────┐
         │   Evaluation & Reports      │  │    Model Artifact Store     │
         │   src/evaluate_model.py     │  │    models/churn_model.pkl   │
         │   ├── model_metrics.json    │  │    (Full sklearn Pipeline)  │
         │   └── feature_importance.png│  └──────────────┬──────────────┘
         └─────────────────────────────┘                 │
                                            ┌────────────┴────────────┐
                                            ▼                         ▼
                              ┌─────────────────────┐  ┌─────────────────────────┐
                              │  Streamlit App       │  │  Monitoring Dashboard   │
                              │  app/streamlit_app.py│  │  monitoring/dashboard.py │
                              │  Port: 8501          │  │  Port: 8502             │
                              └─────────────────────┘  └─────────────────────────┘
```

---

## 📂 Project Folder Structure

```
customer-churn-prediction/
│
├── 📄 README.md                         # Project documentation (you are here)
├── 📄 requirements.txt                  # Python package dependencies
├── 🐳 Dockerfile                        # Container image for deployment
├── 📄 .gitignore                        # Ignored files for version control
├── 📄 LICENSE                           # MIT Open Source License
│
├── 📁 data/
│   ├── raw/
│   │   └── telco_customer_churn.csv     # Original Kaggle dataset (IBM Telco)
│   └── processed/
│       └── cleaned_data.csv             # Cleaned, memory-optimized dataset
│
├── 📓 notebooks/
│   ├── 01_data_exploration.ipynb        # EDA: distributions, class balance, correlations
│   ├── 02_feature_engineering.ipynb     # Pipeline construction and validation
│   └── 03_model_training.ipynb          # Training, tuning, benchmark reports
│
├── 🐍 src/
│   ├── data_preprocessing.py            # Data loading, cleaning, type optimization
│   ├── feature_engineering.py           # sklearn Pipeline + ColumnTransformer
│   ├── train_model.py                   # Model training + artifact serialization
│   ├── evaluate_model.py                # Metrics, performance plots, JSON reports
│   └── predict.py                       # Inference module for production use
│
├── 📦 models/
│   └── churn_model.pkl                  # Serialized sklearn Pipeline (Preprocessor + XGBoost)
│
├── 🖥️  app/
│   └── streamlit_app.py                 # Interactive churn prediction dashboard
│
├── 📡 monitoring/
│   └── model_performance_dashboard.py   # Static metrics + feature importance dashboard
│
├── 📊 reports/
│   ├── model_metrics.json               # Accuracy, F1, ROC-AUC snapshot
│   └── feature_importance.png           # Top-10 XGBoost feature importances
│
└── 🧪 tests/
    └── test_data_pipeline.py            # pytest integration tests for data pipeline
```

---

## 📋 Dataset Description

| Property | Detail |
|---|---|
| **Source** | [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Records** | 7,043 customers |
| **Features** | 21 (demographics, account info, service subscriptions) |
| **Target** | `Churn` — Binary (Yes / No) |
| **Class Balance** | ~73.5% No Churn / 26.5% Churn (imbalanced) |
| **Format** | CSV |

### Feature Groups

| Category | Features |
|---|---|
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Account** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| **Services** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Target** | `Churn` |

---

## 🔬 ML Pipeline Explanation

The system is built around a **Scikit-Learn Pipeline architecture** to prevent data leakage and ensure reproducibility across training and inference.

```python
# Numerical Features
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler())                    # Normalize scale
])

# Categorical Features
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combined Transformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])

# Full End-to-End Pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(...))
])
```

> **Why Pipelines?** Fitting transformers (like `StandardScaler`) on the full dataset and then using those fitted scalers on the test set constitutes **data leakage**. Scikit-Learn Pipelines ensure all transformations are exclusively fitted on training data, guaranteeing honest performance estimates.

---

## 🏋️ Model Training Process

Three models were benchmarked using consistent preprocessing via the shared Pipeline:

| Model | Rationale |
|---|---|
| **Logistic Regression** | Fast, interpretable baseline; good for linearly separable problems |
| **Random Forest** | Strong ensemble learner; robust to outliers and noisy features |
| **XGBoost** | State-of-the-art gradient boosting; optimized for tabular data |

### Training Configuration

```python
# Data Split
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Stratification ensures class balance (26% churn) is maintained in both splits

# Models
LogisticRegression(max_iter=1000, random_state=42)
RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
XGBClassifier(eval_metric='logloss', n_jobs=-1, random_state=42)
```

### Reproducibility

All experiments are fully reproducible using:

```bash
python src/train_model.py   # Trains model with random_state=42
python src/evaluate_model.py # Generates metrics JSON + importance plot
```

Model artifacts are versioned as `.pkl` files in `models/`. For full experiment tracking, this project can be extended with **MLflow** or **Weights & Biases** (see [Future Improvements](#-future-improvements)).

---

## 📊 Model Evaluation Metrics

All metrics are evaluated on a **stratified 20% held-out test set** never seen during training.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 80.6% | 0.657 | 0.559 | 0.604 | 0.842 |
| **XGBoost** | 78.5% | 0.606 | 0.543 | 0.573 | 0.825 |
| **Random Forest** | 77.9% | 0.605 | 0.476 | 0.533 | 0.817 |

> **Metric of Choice: F1-Score** — In a churn prediction context, both false positives (wasted retention spend) and false negatives (lost revenue from undetected churners) have business costs. F1-Score balances precision and recall, making it the target optimization metric.

> **ROC-AUC Interpretation**: A score of 0.84 means the model correctly distinguishes a churner from a non-churner 84% of the time — significantly better than random (0.5).

### Performance Snapshot (from `reports/model_metrics.json`)

```json
{
    "accuracy": 0.812,
    "precision": 0.657,
    "recall": 0.559,
    "f1_score": 0.604,
    "roc_auc": 0.841
}
```

> This file is auto-generated on every run of `python src/evaluate_model.py`.

---

## 🧠 Explainable AI (SHAP)

**SHAP (SHapley Additive exPlanations)** provides mathematically rigorous, model-agnostic explanations for both individual predictions and global model behavior. This is critical for business stakeholders making retention decisions.

### Global Feature Importance

The SHAP global importance plot (`reports/feature_importance.png`) reveals the **most influential features across all customers:**

| Rank | Feature | Business Insight |
|:---:|---|---|
| 🥇 | `tenure` | Long-tenured customers are far less likely to churn |
| 🥈 | `Contract` | Month-to-month contracts are the strongest churn signal |
| 🥉 | `MonthlyCharges` | Higher bills correlate with churn — pricing sensitivity |
| 4 | `TotalCharges` | Proxy for customer lifetime value |
| 5 | `InternetService` | Fiber optic customers churn more than DSL customers |

### Individual (Local) Explanation

For every prediction made in the dashboard, a **SHAP Waterfall Plot** shows exactly which features pushed the probability up or down for that specific customer:

```
E[f(x)] = 0.273 (Base Rate)
 ─────────────────────────────────────────
  + Contract=Month-to-month: +0.183  🔴
  + tenure=6:                +0.142  🔴
  + MonthlyCharges=$85:      +0.091  🔴
  - TechSupport=Yes:         -0.063  🔵
  - OnlineSecurity=Yes:      -0.051  🔵
 ─────────────────────────────────────────
f(x) = 0.743 → ⚠️ HIGH CHURN RISK
```

---

## 🖥️ Streamlit Application Overview

The project ships with **two interactive Streamlit dashboards**:

### 1. Prediction Dashboard (`app/streamlit_app.py`)
>
> **Port:** `8501` | **Command:** `streamlit run app/streamlit_app.py`

**Tab 1 — Individual Prediction:**

- Customer attribute sidebar (tenure, contract, charges, internet service)
- Real-time churn probability with risk tiers (Low / Medium / High)
- SHAP Waterfall Plot explaining every prediction

**Tab 2 — Global Insights & Monitoring:**

- Top 10 Feature Importance bar chart (XGBoost internal scores)
- Model performance metrics snapshot
- Model health status indicator

### 2. Model Monitoring Dashboard (`monitoring/model_performance_dashboard.py`)
>
> **Port:** `8502` | **Command:** `streamlit run monitoring/model_performance_dashboard.py`

- Reads live `reports/model_metrics.json`
- Displays Accuracy, Precision, Recall, F1-Score, ROC-AUC as KPI cards
- Renders `reports/feature_importance.png` from last evaluation run
- Designed for data science teams tracking model health over time

---

## 🚀 Installation Guide

### Prerequisites

- Python 3.10+
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

### Step 2: Set Up the Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### 1. Preprocess Data

```bash
python src/data_preprocessing.py
```

> Reads from `data/raw/telco_customer_churn.csv` → Writes to `data/processed/cleaned_data.csv`

### 2. Train the Model

```bash
python src/train_model.py
```

> Trains XGBoost pipeline → Saves to `models/churn_model.pkl`

### 3. Evaluate the Model

```bash
python src/evaluate_model.py
```

> Generates `reports/model_metrics.json` and `reports/feature_importance.png`

### 4. Run Automated Tests

```bash
pytest tests/test_data_pipeline.py -v
```

> All tests should report `PASSED`

### 5. Launch Prediction Dashboard

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

> Open: [http://localhost:8501](http://localhost:8501)

### 6. Launch Monitoring Dashboard

```bash
streamlit run monitoring/model_performance_dashboard.py --server.port 8502
```

> Open: [http://localhost:8502](http://localhost:8502)

---

## 🐳 Docker Deployment

Deploy the entire system in a single command using Docker:

### Build the Image

```bash
docker build -t churn-prediction-system .
```

### Run the Container

```bash
docker run -p 8501:8501 churn-prediction-system
```

### Access the App

Open your browser at: **[http://localhost:8501](http://localhost:8501)**

### Production Deployment (Cloud)

The Docker container is compatible with:

| Platform | Deployment Method |
|---|---|
| **AWS ECS / Fargate** | Push to ECR → Deploy via task definition |
| **Google Cloud Run** | `gcloud run deploy --source . --port 8501` |
| **Azure Container Apps** | `az containerapp up --source .` |
| **Kubernetes** | Apply `deployment.yaml` with image reference |

```bash
# Example: Push to Docker Hub and run
docker tag churn-prediction-system your-username/churn-prediction-system
docker push your-username/churn-prediction-system

# Pull and run anywhere
docker run -p 8501:8501 your-username/churn-prediction-system
```

---

## 🔭 Future Improvements

The following enhancements would elevate this project to a full-scale MLOps platform:

| Enhancement | Description | Priority |
|---|---|:---:|
| 🧪 **MLflow Integration** | Track all experiments, hyperparameters, and model versions centrally | High |
| ⚙️ **Hyperparameter Tuning** | Implement `Optuna` or `GridSearchCV` for systematic optimization | High |
| 📉 **Class Imbalance Handling** | Apply `SMOTE`, class weighting, or threshold optimization for better recall | High |
| 🔄 **CI/CD Pipeline** | GitHub Actions for automated testing, model retraining, and deployment | Medium |
| 🌊 **Data Drift Detection** | Integrate `evidently` or `deepchecks` to alert when data distribution shifts | Medium |
| ⏱️ **Real-Time Feature Store** | Replace static CSV input with a live feature store (e.g., Feast, Tecton) | Medium |
| 🧬 **Deep Learning Baseline** | TabNet or a simple MLP as a comparison to tree-based models | Low |
| 📱 **REST API (FastAPI)** | Expose model as a production API with `/predict` endpoint | High |
| 🗃️ **Database Integration** | Store prediction logs in PostgreSQL for downstream analysis | Medium |

---

## 💡 Key Learnings

This project demonstrates mastery of the following production ML engineering concepts:

- **Pipeline Architecture**: Using `sklearn.Pipeline` + `ColumnTransformer` to create robust, leakage-free preprocessing that seamlessly handles both training and inference.
- **Explainability in Production**: Integrating SHAP into a live dashboard to surface both global model behavior and individual prediction reasoning.
- **Modular Code Design**: Separating concerns into discrete, importable, and testable Python modules — enabling independent development and testing of each component.
- **Reproducibility**: Seeded random states, documented training commands, and serialized artifacts ensure the exact same results are reproducible by any team member.
- **Containerization**: Packaging the entire ML system (model, preprocessor, and UI) into a portable Docker image ready for cloud deployment.
- **Model Monitoring**: Building a dedicated monitoring dashboard that reads from auto-generated metric reports after each evaluation cycle.
- **Testing Culture**: Writing `pytest` unit tests for data pipeline validation ensures regressions are caught before they propagate to model training.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by a Ishan Shirode**

*If you find this project useful, please ⭐ star the repository — it helps others discover it!*

[![GitHub Stars](https://img.shields.io/github/stars/your-username/customer-churn-prediction?style=social)](https://github.com/your-username/customer-churn-prediction)

</div>
