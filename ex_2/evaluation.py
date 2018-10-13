import logging

from ex_2 import CORRECT_LABEL, INCORRECT_LABEL


logger = logging.getLogger(__name__)

tp = 0
fp = 0
tn = 0
fn = 0


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


def stats():
    logger.info("TP: {}, FP: {}, TN: {}, FN: {}".format(tp, fp, tn, fn))
    logger.info("Accuracy: {}".format(1 - (fp + fn)/(tp + fp + tn + fn)))
    logger.info("Recall: {}".format(tp/(tp + fn)))
    logger.info("Precision: {}".format(tp/(tp + fp)))
