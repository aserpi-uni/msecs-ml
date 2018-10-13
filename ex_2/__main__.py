import logging

from ex_2 import ALGORITHM, CORRECT_LABEL, INCORRECT_LABEL
from ex_2.data_preprocessing import partition_dataset

logger = logging.getLogger(__name__)

corrects, incorrects, correct_dict, incorrect_dict = partition_dataset()

logger.info("Sample ({}/{}): {}/{}".format(CORRECT_LABEL, INCORRECT_LABEL, corrects, incorrects))

ALGORITHM.learn(corrects, incorrects, correct_dict, incorrect_dict)
