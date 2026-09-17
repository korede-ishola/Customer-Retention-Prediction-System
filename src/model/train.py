from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

def train_model(X_train, y_train, data_processor, model_params):
    """
    Transforms the preprocessed data and trains the model

    """

    # Create model
    xgb = XGBClassifier(**model_params)

    # Create Pipeline
    inference_pipeline = Pipeline(steps=[
        ('data_processor', data_processor),
        ('model', xgb)
    ])

    # Train_model
    inference_pipeline.fit(X_train, y_train)

    return inference_pipeline