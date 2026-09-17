import pandas as pd

def preprocess_data(df: pd.DataFrame, cat_features: list, num_features: list, target_col: str='churn') -> pd.DataFrame:
    """
    Preprocessing Telco Data:
    i. Drop irrelevant columns
    ii. Encode target column
    iii. Normalize inputs and headers
    
    """

    # Normalize df headers
    df.columns = (
      df.columns.astype(str)
      .str.strip()
      .str.lower()
      .str.replace(r"\s+", "", regex=True)
    )

    # Normalize categorical inputs
    for col in cat_features:
        if df[col].dtype == "object":
            df[col] = df[col].str.lower()

    # Normalize numerical inputs
    for col in num_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop irrelevant columns
    if "customerid" in df.columns:
        df = df.drop('customerid', axis=1)

    # Encode target column
    if target_col in df.columns:
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    return df 