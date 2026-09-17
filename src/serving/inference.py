import pandas as pd
import joblib

from config import config
from src.data.preprocess_data import preprocess_data
from src.data.validate_data import validate_data

model = joblib.load(config.MODEL_PATH)

def make_prediction(customer_data: dict):
    """
    Makes prediction for given customer data.
    Return model's prediction and it's interpretation.
    
    """

    df = pd.DataFrame([customer_data])

    # Same preprocessing used for the training data
    df = preprocess_data(df, config.CATEGORICAL_FEATURES, config.NUMERICAL_FEATURES)

    # Validate data
    validated, failed = validate_data(df, config.FEATURES, config.NUMERICAL_FEATURES)
    if not validated:
        raise ValueError(f"Data validation failed. Issues: {failed}")

    # Inference
    y = (model.predict_proba(df)[:, 1] > config.THRESHOLD).astype(int)
    
    prediction = "Likely to churn" if y == 1 else "Not likely to churn"

    return y, prediction