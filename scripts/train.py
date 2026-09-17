import mlflow

from config import config

from src.data.load_data import load_data
from src.data.save_data import save_data
from src.data.preprocess_data import preprocess_data
from src.data.validate_data import validate_data

from src.features.transform_pipeline import transform_pipeline

from src.model.train import train_model
from src.model.evaluate import evaluate_model
from src.model.tune import tune_model

from src.utils.split_data import split_data

def main():
    """ Orchestrates and runs the training pipeline. """

    mlflow.set_tracking_uri(config.EXPERIMENT_FOLDER)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    # Load, preprocess and validate data
    df = load_data(config.DATA_PATH)

    df = preprocess_data(df, config.CATEGORICAL_FEATURES, config.NUMERICAL_FEATURES, config.TARGET)
    save_data(df, config.PROCESSED_DATA_PATH)

    validated, failed = validate_data(df, config.FEATURES, config.NUMERICAL_FEATURES)
    if not validated:
        raise ValueError(f"Data validation failed. Issues: {failed}")

    # Create training and test sets
    X = df.drop('churn', axis=1)
    y = df['churn']

    X_train, X_test, y_train, y_test = split_data(X, y, config.TEST_SIZE, config.RANDOM_STATE)

    # Load data transform pipeline
    data_processor = transform_pipeline(config.CATEGORICAL_FEATURES, config.NUMERICAL_FEATURES)

    # Tune model parameters
    model_params = tune_model(
        X_train, y_train, data_processor, 
        config.TEST_SIZE, config.THRESHOLD, config.RANDOM_STATE, 
        config.FN_FP_COST_RATIO
        )

    # Train and evaluate model
    with mlflow.start_run():
        model = train_model(X_train, y_train, data_processor, model_params)
        precision, recall, cost = evaluate_model(
            model, X_test, y_test, config.THRESHOLD, config.FN_FP_COST_RATIO
            )

        # Log parameters and model
        mlflow.log_params(model_params)
        mlflow.log_params({
            'model': 'XGBoost',
            'test_size': config.TEST_SIZE,
            'random_state': config.RANDOM_STATE,
        })
        mlflow.log_metrics({
            "precision": precision,
            "recall": recall,
            "cost": cost
        })
        mlflow.sklearn.log_model(
            model,
            name='telco_churn_pipeline'
            )

if __name__ == "__main__":
    main()