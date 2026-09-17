from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def transform_pipeline(cat_features: list, num_features: list):
    """
    To transform preprocessed data into model-ready data
    Fill missing values
    Scale numerical features
    Encode categorical features

    """

    # Scale numerical features and fill null values
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Encode categorical features and fill null values
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create processing pipeline
    processor = ColumnTransformer([
        ('numerical', numerical_pipeline, num_features),
        ('categorical', categorical_pipeline, cat_features)
    ])

    return processor
