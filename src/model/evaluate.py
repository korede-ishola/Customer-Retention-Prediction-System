from sklearn.metrics import precision_score, recall_score, classification_report

from src.utils.business_cost import business_cost

def evaluate_model(model, X_test, y_test, threshold, fn_fp_cost_ratio):
    """
    Evaluates a model.
    Returns precsion, recall, and business cost.

    """

    # Make prediction
    y_pred = model.predict_proba(X_test)[:, 1] > threshold

    # Evaluate model
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cost = business_cost(y_test, y_pred, fn_fp_cost_ratio)

    return precision, recall, cost