import warnings
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score

def f2(preds, targs, start=0.17, end=0.24, step=0.01):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return max([fbeta_score(targs, (preds>th), 2, average='samples')
                    for th in np.arange(start,end,step)])


def average_precision(preds, targs, threshold=0.5, num_classes=7):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sum([precision_score(targs[:,i], preds[:,i] > threshold) for i in range(num_classes)]) / num_classes

    
def average_recall(preds, targs, threshold=0.5, num_classes=7):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sum([recall_score(targs[:,i], preds[:,i] > threshold) for i in range(num_classes)]) / num_classes
    
    
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