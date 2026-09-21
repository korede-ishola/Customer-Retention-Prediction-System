import pandas as pd

from src.data.preprocess_data import preprocess_data
from src.data.validate_data import validate_data

from config import config

CAT_FEATURES = config.CATEGORICAL_FEATURES
NUM_FEATURES = config.NUMERICAL_FEATURES
FEATURES = config.FEATURES


def test_preprocess_data_normalizes_columns_and_values():
    df = pd.DataFrame({
        " Customer ID ": ["001"],
        "Gender": ["Male"],
        "Partner": [" YES "],
        "Dependents": ["No"],
        "Tenure": ["12"],
        "Monthly Charges": ["50.5"],
        "Total Charges": ["606"],
        "Churn": ["Yes"],
    })

    result = preprocess_data(
        df,
        CAT_FEATURES,
        NUM_FEATURES
    )

    assert "customerid" not in result.columns
    assert "monthlycharges" in result.columns
    assert "totalcharges" in result.columns

    assert result["gender"].iloc[0] == "male"
    assert result["partner"].iloc[0] == " yes "
    assert result["tenure"].iloc[0] == 12
    assert result["monthlycharges"].iloc[0] == 50.5
    assert result["churn"].iloc[0] == 1


def test_preprocess_data_encodes_churn():
    df = pd.DataFrame({
        "Gender": ["Male", "Female"],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "Tenure": ["12", "24"],
        "MonthlyCharges": ["50", "70"],
        "TotalCharges": ["600", "1680"],
        "Churn": ["Yes", "No"],
    })

    result = preprocess_data(
        df,
        CAT_FEATURES,
        NUM_FEATURES
    )

    assert result["churn"].tolist() == [1, 0]


def test_validate_data_accepts_valid_data():
    df = pd.DataFrame({
        "gender": ["male", "female"],
        "partner": ["yes", "no"],
        "dependents": ["no", "yes"],
        "tenure": [12, 24],
        "monthlycharges": [50.0, 70.0],
        "totalcharges": [600.0, 1680.0],
    })

    valid, failed = validate_data(
        df,
        FEATURES,
        NUM_FEATURES
    )

    assert valid is True
    assert failed == []


def test_validate_data_detects_missing_columns():
    df = pd.DataFrame({
        "gender": ["male"],
        "partner": ["yes"],
        "tenure": [12],
    })

    valid, failed = validate_data(
        df,
        FEATURES,
        NUM_FEATURES
    )

    assert valid is False
    assert any("Missing columns" in error for error in failed)


def test_validate_data_detects_negative_values():
    df = pd.DataFrame({
        "gender": ["male"],
        "partner": ["yes"],
        "dependents": ["no"],
        "tenure": [-1],
        "monthlycharges": [50.0],
        "totalcharges": [600.0],
    })

    valid, failed = validate_data(
        df,
        FEATURES,
        NUM_FEATURES
    )

    assert valid is False
    assert any("contains negative values" in error for error in failed)