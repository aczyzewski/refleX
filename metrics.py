# coding: utf-8

import numpy as np
from keras import backend as K

__author__ = "Dariusz Brzezinski"


def hamming_score(y_true, y_pred):
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

def hamming_loss(y_true, y_pred):
    return K.sum(K.abs(y_true - K.round(y_pred))) / K.cast(K.shape(y_true)[0] * K.shape(y_true)[1], K.floatx())
