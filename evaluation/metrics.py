import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

def evaluate_predictions(y_true, y_pred):
    """Compute evaluation metrics for Diabetic Retinopathy grading."""
    acc = accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    metrics = {
        'accuracy': float(acc),
        'quadratic_weighted_kappa': float(qwk),
        'f1_macro': float(f1_macro)
    }
    return metrics
