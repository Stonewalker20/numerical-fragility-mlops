import numpy as np

def prediction_variance(predictions_list):
    """
    predictions_list: list of np arrays with shape (N, C) (logits or probs)
    Returns mean variance over all elements.
    """
    stack = np.stack(predictions_list, axis=0)  # (R, N, C)
    return float(np.var(stack, axis=0).mean())

def disagreement_rate(pred_a, pred_b):
    """
    pred_a, pred_b: np arrays of predicted class indices shape (N,)
    """
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    return float((pred_a != pred_b).mean())
