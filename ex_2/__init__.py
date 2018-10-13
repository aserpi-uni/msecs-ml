import logging
from os import getenv


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_algorithm():
    from ex_2.algorithm import Algorithm

    global ALGORITHM
    ALGORITHM = Algorithm(getenv("ALGORITHM", "multinomial"))


INPUT_FILE = getenv("INPUT_FILE", "./ex_2/SMSSpamCollection")
DATASET = getenv("DATASET_FILE", "./ex_2/dataset")
CORRECT_LABEL = getenv("CORRECT_LABEL", "ham")
INCORRECT_LABEL = getenv("INCORRECT_LABEL", "spam")

set_algorithm()
LAPLACE_SMOOTHING = not (getenv("LAPLACE_SMOOTHING") == "false")
SAMPLE_RATE = float(getenv("SAMPLE_RATE", 0.3))

logger.info(
    "Input file: {}\nDataset: {}\nLabel (correct/incorrect): {}".format(INPUT_FILE, DATASET, CORRECT_LABEL,
                                                                        INCORRECT_LABEL, SAMPLE_RATE))
logger.info("Algorithm: {}\nSample rate: {}, Laplace smoothing: {}".format(ALGORITHM, SAMPLE_RATE, LAPLACE_SMOOTHING))

