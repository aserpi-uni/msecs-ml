from ex_2 import *
from ex_2.sample import create_sample

logger = logging.getLogger(__name__)

sample = create_sample(FILE, SAMPLE_RATE, CORRECT_LABEL, INCORRECT_LABEL)
