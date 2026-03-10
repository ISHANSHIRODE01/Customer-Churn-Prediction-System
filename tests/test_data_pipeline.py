import pytest
import pandas as pd
import sys
import os

# Add src to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_preprocessing import clean_data

def test_data_cleaning():
    """Verify numeric columns are cleaned and customerID is removed."""
    # Build a small dummy dataset
    test_df = pd.DataFrame([{
        'customerID': '123-ABC',
        'TotalCharges': '100.5',
        'InternetService': 'DSL'
    }])
    
    cleaned_df = clean_data(test_df)
    
    # Assertions
    assert 'customerID' not in cleaned_df.columns
    assert pd.api.types.is_numeric_dtype(cleaned_df['TotalCharges'])
    assert cleaned_df['TotalCharges'][0] == 100.5

if __name__ == "__main__":
    test_data_cleaning()
    print("Tests Passed Correctly.")
