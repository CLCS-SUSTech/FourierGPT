import numpy as np
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from run_pwh_cls import SpectrumData


def get_power_thresholds(sp_human: SpectrumData, sp_model: SpectrumData, k_threshold: int, n_intervals: int):
    human_power = sp_human.spectrum_df.groupby('sid').head(k_threshold)['power']
    model_power = sp_model.spectrum_df.groupby('sid').head(k_threshold)['power']
    human_mean = human_power.mean()
    model_mean = model_power.mean()
    higher = 'model' if human_mean < model_mean else 'human'

    low = min(human_power.min(), model_power.min())
    high = max(human_power.max(), model_power.max())
    mid = (low + high) / 2
    low_thresholds = np.linspace(low, mid, n_intervals)
    high_thresholds = np.linspace(mid, high, n_intervals)
    power_thresholds = np.concatenate([low_thresholds[:-1], high_thresholds])

    return power_thresholds, higher

def classify(x, k_threshold, power_threshold, heuristics='>'):
    preds = 0
    for sid in x.keys():
        power = x[sid]['power']
        if heuristics == '>':
            if np.mean(power[:k_threshold]) > power_threshold:
                preds += 1
        elif heuristics == '<':
            if np.mean(power[:k_threshold]) < power_threshold:
                preds += 1
    return preds, len(x)
        

def get_roc(sp_human, sp_model, k_threshold, n_intervals=5):
    """
    Positive: model-generated
    Negative: human-written
    ROC curve is TPR vs. FPR
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    """
    roc_results = []
    power_thresholds, higher = get_power_thresholds(sp_human, sp_model, k_threshold, n_intervals=n_intervals)
    x_human = sp_human.get_dict()
    x_model = sp_model.get_dict()

    for power_threshold in power_thresholds:
        if higher == 'model':
            TP, P = classify(x_model, k_threshold, power_threshold, heuristics='>')
            FP, N = classify(x_human, k_threshold, power_threshold, heuristics='>')
        elif higher == 'human':
            TP, P = classify(x_model, k_threshold, power_threshold, heuristics='<')
            FP, N = classify(x_human, k_threshold, power_threshold, heuristics='<')
        TPR = TP / P
        FPR = FP / N
        acc = (TP + N - FP) / (P + N)
        roc_results.append((TPR, FPR, acc))

    return sorted(roc_results, key=lambda x: x[1]) 