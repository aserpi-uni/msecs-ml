import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_FILE = "./ex_2/SMSSpamCollection"
DATASET_FILE = "./ex_2/dataset"
SAMPLE_RATE = 0.1


CORRECT_LABEL = "ham"
INCORRECT_LABEL = "spam"

logger.info(
    "Impot file: {}\nDataset: {}<nCorrect label: {}\nIncorrect label: {}\nSample rate: {}".format(INPUT_FILE,
                                                                                                  DATASET_FILE,
                                                                                                  CORRECT_LABEL,
                                                                                                  INCORRECT_LABEL,
                                                                                                  SAMPLE_RATE))
