import pytest
import pandas as pd
from credit_risk.processing import preprocess_credit_data

def test_column_renaming():
    # Create a minimal raw df with a few columns from COLUMN_NAMES
    raw_data = {
        "laufkont": [1, 2],
        "laufzeit": [12, 24],
        "kredit": [0, 1]
    }
    df = pd.DataFrame(raw_data)
    processed = preprocess_credit_data(df)
    
    assert "checking_status" in processed.columns
    assert "duration_months" in processed.columns
    assert "credit_risk" not in processed.columns
    assert "bad_credit" in processed.columns

def test_category_mapping():
    raw_data = {
        "laufkont": [1, 4],
        "kredit": [0, 1]
    }
    df = pd.DataFrame(raw_data)
    processed = preprocess_credit_data(df)
    
    # 1: "No account", 4: "200+ DM"
    assert processed["checking_status"].iloc[0] == "No account"
    assert processed["checking_status"].iloc[1] == "200+ DM"

def test_target_transformation():
    raw_data = {
        "kredit": [0, 1, 0] # 0=bad, 1=good
    }
    df = pd.DataFrame(raw_data)
    processed = preprocess_credit_data(df)
    
    # bad_credit = (credit_risk == 0)
    assert processed["bad_credit"].tolist() == [1, 0, 1]

def test_unexpected_codes_raise_error():
    raw_data = {
        "laufkont": [99], # 99 is not in CATEGORY_LABELS["checking_status"]
        "kredit": [0]
    }
    df = pd.DataFrame(raw_data)
    with pytest.raises(ValueError, match="Unexpected codes in checking_status"):
        preprocess_credit_data(df)

def test_missing_target_raises_error():
    raw_data = {
        "laufkont": [1]
        # missing 'kredit'
    }
    df = pd.DataFrame(raw_data)
    with pytest.raises(KeyError, match="Column 'credit_risk' missing"):
        preprocess_credit_data(df)

def test_invalid_target_values_raise_error():
    raw_data = {
        "kredit": [0, 2] # 2 is invalid
    }
    df = pd.DataFrame(raw_data)
    with pytest.raises(ValueError, match="Unexpected target values in 'credit_risk'"):
        preprocess_credit_data(df)
