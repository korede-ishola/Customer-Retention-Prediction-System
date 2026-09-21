import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model.train import train_model


def test_train_model_returns_fitted_pipeline():

    X_train = pd.DataFrame({
        "gender": ["male", "female", "male", "female", "male", "female"],
        "tenure": [1, 12, 24, 36, 48, 60],
        "monthlycharges": [30, 50, 60, 70, 80, 90],
    })

    y_train = [0, 1, 0, 1, 0, 1]

    categorical_features = ["gender"]
    numerical_features = ["tenure", "monthlycharges"]

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    data_processor = ColumnTransformer([
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        )
    ])

    model_params = {
        "n_estimators": 10,
        "max_depth": 2,
        "learning_rate": 0.1,
        "random_state": 42,
        "eval_metric": "logloss",
    }

    model = train_model(
        X_train,
        y_train,
        data_processor,
        model_params
    )

    assert isinstance(model, Pipeline)
    assert "data_processor" in model.named_steps
    assert "model" in model.named_steps

    predictions = model.predict(X_train)

    assert len(predictions) == len(X_train)
    assert set(predictions).issubset({0, 1})