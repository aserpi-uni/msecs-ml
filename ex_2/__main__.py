import logging

from ex_2 import ALGORITHM, CORRECT_LABEL, INCORRECT_LABEL, ITERATIONS
from ex_2.data_preprocessing import partition_dataset
from ex_2.evaluation import compile_stats, get_total_stats

logger = logging.getLogger(__name__)

accuracy = 0
recall = 0
precision = 0
for i in range(1, ITERATIONS):
    corrects, incorrects, correct_dict, incorrect_dict = partition_dataset()

    logger.info("Sample ({}/{}): {}/{}".format(CORRECT_LABEL, INCORRECT_LABEL, corrects, incorrects))

    ALGORITHM.learn(corrects, incorrects, correct_dict, incorrect_dict)

    statistics = compile_stats()

    logger.info("TP: {}, FP: {}, TN: {}, FN: {}".format(statistics["total"]["TP"], statistics["total"]["FP"],
                                                        statistics["total"]["TN"], statistics["total"]["FN"],))
    logger.info("Accuracy: {}".format(statistics["accuracy"]))
    logger.info("Recall: {}".format(statistics["recall"]))
    logger.info("Precision: {}".format(statistics["precision"]))

statistics = get_total_stats()
logger.info("\n\n\n=================\nTOTAL")
logger.info("TP: {}, FP: {}, TN: {}, FN: {}".format(statistics["total"]["TP"], statistics["total"]["FP"],
                                                    statistics["total"]["TN"], statistics["total"]["FN"],))
logger.info("Accuracy: {}".format(statistics["accuracy"]))
logger.info("Recall: {}".format(statistics["recall"]))
logger.info("Precision: {}".format(statistics["precision"]))
