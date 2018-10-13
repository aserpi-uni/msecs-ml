import logging

from ex_2 import CORRECT_LABEL, INCORRECT_LABEL


logger = logging.getLogger(__name__)
iterations = 0
total_tp = 0
total_fp = 0
total_tn = 0
total_fn = 0
total_accuracy = 0
total_recall = 0
total_precision = 0


def evaluate(head, correct_prob, incorrect_prob):
    global tp, fp, tn, fn

    if correct_prob >= incorrect_prob:
        if head == CORRECT_LABEL:
            tp += 1
            return 'TP'
        else:
            fp += 1
            return 'FP'
    else:
        if head == INCORRECT_LABEL:
            tn += 1
            return 'TN'
        else:
            fn += 1
            return 'FN'


def initialize_stats():
    global tp, fp, tn, fn
    tp = 0
    fp = 0
    tn = 0
    fn = 0


def compile_stats():
    global iterations, total_tp, total_fp, total_tn, total_fn, total_accuracy, total_recall, total_precision
    iterations += 1
    total_tp += tp
    total_fp += fp
    total_tn += tn
    total_fn += fn

    accuracy = 1 - (fp + fn)/(tp + fp + tn + fn)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)

    total_accuracy += accuracy
    total_recall += recall
    total_precision += precision

    return {
        "total": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision}


def get_total_stats():
    return {
        "total": {"TP": total_tp, "FP": total_fp, "TN": total_tn, "FN": total_fn},
        "accuracy": total_accuracy/iterations,
        "recall": total_recall/iterations,
        "precision": total_precision/iterations}
