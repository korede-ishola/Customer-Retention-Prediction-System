import optuna
from xgboost import XGBClassifier

from src.features.transform_data import transform_data
from src.utils.split_data import split_data
from src.utils.business_cost import business_cost

def tune_model(X, y, processor, test_size, threshold, fn_fp_cost_ratio, random_state):
    """ Finds the best parameters for the model using Optuna. """

    X = transform_data(X, processor)
    X_train, X_val, y_train, y_val = split_data(X, y, test_size, random_state)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        params = {
            "random_state": random_state,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "eval_metric": "logloss"
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:,1]
        y_pred = (proba >= threshold).astype(int)

        return business_cost(y_val, y_pred, fn_fp_cost_ratio)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    return study.best_params