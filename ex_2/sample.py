import logging
from random import random

logger = logging.getLogger(__name__)


def count_samples(sample, correct_label, incorrect_label):
    corrects = 0
    incorrects = 0

    for msg in sample:
        head = msg.split("\t")[0]
        if head == correct_label:
            corrects += 1
        elif head == incorrect_label:
            incorrects += 1
        else:
            raise ValueError("Malformed file")

    return corrects, incorrects


def create_sample(file, sample_rate, correct_label, incorrect_label):
    while True:
        sample = select_sample(file, sample_rate)
        corrects, incorrects = count_samples(sample, correct_label, incorrect_label)

        if corrects != 0 and incorrects != 0:
            logger.info("Corrects: {}, incorrects: {}".format(corrects, incorrects))
            break
        else:
            logger.info("Sample aborted")


def select_sample(file, sample_rate):
    fin = open(file)
    sample = []

    line = fin.readline()
    while line:
        if random() < sample_rate:
            sample.append(line)
        line = fin.readline()

    fin.close()
    return sample
