import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILE = "./ex_2/SMSSpamCollection"
SAMPLE_RATE = 0.1


CORRECT_LABEL = "ham"
INCORRECT_LABEL = "spam"

logger.info("Dataset: {}\nCorrect label: {}\nIncorrect label: {}\nSample rate: {}".format(FILE, CORRECT_LABEL,
                                                                                          INCORRECT_LABEL, SAMPLE_RATE))
