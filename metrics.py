import numpy as np

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
#
# def _keras_hamming_loss(y_true, y_pred):
#     n_differences = count_nonzero(y_true - y_pred)
#     return (n_differences /
#             (y_true.shape[0] * len(labels) * weight_average))