import pandas as pd
from typing import Tuple, List


def validate_data(
    df: pd.DataFrame,
    features: list,
    num_features: list
) -> Tuple[bool, List[str]]:

    failed = []

    # Check required columns
    missing_columns = [
        feature for feature in features
        if feature not in df.columns
    ]

    if missing_columns:
        failed.append(
            f"Missing columns: {missing_columns}"
        )

    # Check numerical columns
    for feature in num_features:
        if feature in df.columns:
            if not pd.api.types.is_numeric_dtype(df[feature]):
                failed.append(
                    f"{feature} is not numeric"
                )

    # Check non-negative numerical values
    for feature in num_features:
        if feature in df.columns:
            if (df[feature] < 0).any():
                failed.append(
                    f"{feature} contains negative values"
                )

    # 4. Check TotalCharges >= MonthlyCharges
    if all(
        col in df.columns
        for col in ["totalcharges", "monthlycharges"]
    ):
        invalid_charges = (
            df["totalcharges"] < df["monthlycharges"]
        ).sum()

        if invalid_charges > len(df) * 0.05:
            failed.append(
                "More than 5% of records have "
                "TotalCharges < MonthlyCharges"
            )

    return len(failed) == 0, failed