import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessing_pipeline(df):
    """Assembles a scikit-learn preprocessing pipeline based on Dataframe types."""
    # Define categorical and numerical features
    num_features = df.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 'float32', 'float64']).columns.tolist()
    if 'Churn' in num_features:
        num_features.remove('Churn')
    if 'SeniorCitizen' in num_features: # It's a flag, potentially categorical or binary
        num_features.remove('SeniorCitizen')
        
    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    if 'Churn' in cat_features:
        cat_features.remove('Churn')
    # Add SeniorCitizen manually back to categorical if treated as one
    cat_features.append('SeniorCitizen')

    # Numerical Transform Pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical Transform Pipeline
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

    return preprocessor

if __name__ == "__main__":
    # Test on processed data
    processed_path = "data/processed/cleaned_data.csv"
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
        pipeline = build_preprocessing_pipeline(df)
        print("Preprocessing pipeline built successfully.")
    else:
        print(f"File {processed_path} not found. Run src/data_preprocessing.py first.")
