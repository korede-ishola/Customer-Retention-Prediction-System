def business_cost(y_true, y_pred, fn_fp_cost_ratio, precision_threshold=0.45, penalty_severity=5000):
    """
    Returns the business cost of a model's output

    """

    # Confusion matrix components
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Base cost
    base_cost = (fn_fp_cost_ratio * fn) + (1.0 * fp)

    # Bottleneck penalty if precision falls below threshold
    penalty = max(0.0, precision_threshold - precision) * penalty_severity

    return base_cost + penalty