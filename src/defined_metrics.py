import warnings
import numpy as np
import os
from sklearn.metrics import fbeta_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix, f1_score, hamming_loss

def get_raw_scores(preds, targs):
    tp, fp, tn, fn = 0, 0, 0, 0

    # Czytelniejszy kod > ładniejszy kod.
    for prediction, target in zip(preds, targs):
        if ((prediction == 1) and (target == 1)): tp += 1
        if ((prediction == 1) and (target == 0)): fp += 1
        if ((prediction == 0) and (target == 0)): tn += 1
        if ((prediction == 0) and (target == 1)): fn += 1

    return (tp, fp, tn, fn)

def f2(preds, targs, start=0.17, end=0.24, step=0.01):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return max([fbeta_score(targs, (preds>th), 2, average='samples') for th in np.arange(start,end,step)])

def macro_average_precision(preds, targs, threshold=0.5, num_classes=7):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sum([precision_score(targs[:,i], preds[:,i] > threshold) for i in range(num_classes)]) / num_classes
    
def macro_average_recall(preds, targs, threshold=0.5, num_classes=7):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sum([recall_score(targs[:,i], preds[:,i] > threshold) for i in range(num_classes)]) / num_classes
    
def macro_mcc(preds, targs, threshold=0.5, num_classes=7):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sum([matthews_corrcoef(targs[:,i], preds[:,i] > threshold) for i in range(num_classes)]) / num_classes

def macro_f1(preds, targs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sum([f1_score(targs[:,i], preds[:,i] > threshold) for i in range(num_classes)]) / num_classes

def hamming_score(y_pred, y_true, threshold=0.5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        y_pred = y_pred > threshold

        acc_list = []
        for i in range(y_true.shape[0]):
            set_true = set(np.where(y_true[i])[0])
            set_pred = set(np.where(y_pred[i])[0])

            if len(set_true) == 0 and len(set_pred) == 0:
                tmp_a = 1
            else:
                tmp_a = len(set_true.intersection(set_pred)) / \
                        float(len(set_true.union(set_pred)))
            acc_list.append(tmp_a)
        return np.mean(acc_list)

def confusion_matrices(preds, targs, threshold=0.5, num_classes=7):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {i: confusion_matrix(targs[:,i], preds[:,i] > threshold) for i in range(num_classes)}


def only_fp(preds, targs, num_classes=7, threshold=0.5):
    results = []
    for class_id in range(num_classes):
        tp, fp, tn, fn = get_raw_scores(targs[:,class_id], preds[:,class_id] > threshold)
        results.append(fp)

    return results

def recall(preds, targs, num_classes=7, threshold=0.5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [recall_score(targs[:,i], preds[:,i] > threshold) for i in range(num_classes)]

def precision(preds, targs, num_classes=7, threshold=0.5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [precision_score(targs[:,i], preds[:,i] > threshold) for i in range(num_classes)]

def exact_match_ratio(preds, targs, threshold=0.5):
    return sum([int(np.array_equal(targs, preds > threshold)) in zip(preds, targs)]) / len(preds)

def hamming_loss_score(preds, targs, num_classes=7, threshold=0.5):
    return (preds != targs).sum() / len(preds) * num_classes

def get_predefined_metrics():
    return [f2, macro_average_precision, macro_average_recall, macro_mcc, hamming_score, exact_match_ratio, hamming_loss_score]

def get_class_specific_metrics():
    return [precision, recall, only_fp]