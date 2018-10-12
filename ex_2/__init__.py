import logging
import random

from ex_2.algorithm import AlgorithmType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_FILE = "./ex_2/SMSSpamCollection"
DATASET_FILE = "./ex_2/dataset"

CORRECT_LABEL = "ham"
INCORRECT_LABEL = "spam"

ALGORITHM = random.choice(list(AlgorithmType))
SAMPLE_RATE = 0.3

logger.info(
    "Input file: {}\nDataset: {}\nCorrect label: {}\nIncorrect label: {}".format(INPUT_FILE, DATASET_FILE,
                                                                                 CORRECT_LABEL, INCORRECT_LABEL,
                                                                                 SAMPLE_RATE))
logger.info("Algorithm: {}\nSample rate: {}".format(ALGORITHM, SAMPLE_RATE))
