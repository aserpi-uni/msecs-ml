import contextlib
import logging
import os
from collections import defaultdict
from random import random

from ex_2 import INPUT_FILE, DATASET_FILE, SAMPLE_RATE, CORRECT_LABEL, INCORRECT_LABEL, ALGORITHM

logger = logging.getLogger(__name__)


def add_words(phrase, dictionary):
    for word in phrase.split(" "):
        dictionary[word] += 1


def clean_line(line):
    head, body = line.split("\t")
    body = body.lower()

    if ALGORITHM.delete_duplicates():
        body = set(body.strip("\n").split(" "))
        body.discard("")
        body = " ".join(body)

    return "{}\t{}\n".format(head, body)


def partition_dataset():
    with contextlib.suppress(FileNotFoundError):
        os.remove(DATASET_FILE)
    corrects = 0
    incorrects = 0
    correct_dict = defaultdict(int)
    incorrect_dict = defaultdict(int)

    with open(INPUT_FILE) as fin, open(DATASET_FILE, "a") as fout:
        for line in fin:
            line = clean_line(line)
            if random() < SAMPLE_RATE:
                if process_sample_line(line, correct_dict, incorrect_dict):
                    corrects += 1
                else:
                    incorrects += 1
            else:
                fout.write("{}".format(line))

    return corrects, incorrects, correct_dict, incorrect_dict


def process_sample_line(line, correct_dict, incorrect_dict):
    head, body = line.strip("\n").split("\t")

    if head == CORRECT_LABEL:
        add_words(body, correct_dict)
        return True
    elif head == INCORRECT_LABEL:
        add_words(body, incorrect_dict)
        return False
    else:
        raise KeyError("Label not recognised")
