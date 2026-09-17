import pandas as pd

def save_data(df: pd.DataFrame, file_path: str):
    """
    Saves a pandas DataFrame to the specified path as CSV.

    """
    
    df.to_csv(file_path, index=False)