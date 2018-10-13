import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_algorithm():
    from ex_2.algorithm import Algorithm

    global ALGORITHM
    ALGORITHM = Algorithm.MULTINOMIAL


INPUT_FILE = "./ex_2/SMSSpamCollection"
DATASET = "./ex_2/dataset"
CORRECT_LABEL = "ham"
INCORRECT_LABEL = "spam"

set_algorithm()
LAPLACE_SMOOTHING = True
SAMPLE_RATE = 0.3

logger.info(
    "Input file: {}\nDataset: {}\nCorrect label: {}\nIncorrect label: {}".format(INPUT_FILE, DATASET,
                                                                                 CORRECT_LABEL, INCORRECT_LABEL,
                                                                                 SAMPLE_RATE))
logger.info("Algorithm: {}\nSample rate: {}, Laplace smoothing; {}".format(ALGORITHM, SAMPLE_RATE, LAPLACE_SMOOTHING))

