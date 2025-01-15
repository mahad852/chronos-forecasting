import numpy as np

def calculate_smape(y_gt, y_pred):
    return np.mean(200 * np.abs(y_pred - y_gt) / (np.abs(y_pred) + np.abs(y_gt) + 1e-8))
