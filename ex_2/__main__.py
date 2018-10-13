import logging

from ex_2 import ALGORITHM, CORRECT_LABEL, INCORRECT_LABEL
from ex_2.data_preprocessing import partition_dataset
from ex_2.evaluation import stats

logger = logging.getLogger(__name__)

corrects, incorrects, correct_dict, incorrect_dict = partition_dataset()

logger.info("Sample ({}/{}): {}/{}".format(CORRECT_LABEL, INCORRECT_LABEL, corrects, incorrects))

ALGORITHM.learn(corrects, incorrects, correct_dict, incorrect_dict)

stats = stats()
logger.info("TP: {}, FP: {}, TN: {}, FN: {}".format(stats["total"]["TP"], stats["total"]["FP"], stats["total"]["TN"],
                                                    stats["total"]["FN"],))
logger.info("Accuracy: {}".format(stats["accuracy"]))
logger.info("Recall: {}".format(stats["recall"]))
logger.info("Precision: {}".format(stats["precision"]))
