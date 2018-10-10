import logging
import os
import string
from collections import defaultdict
from random import random

from ex_2 import INPUT_FILE, DATASET_FILE, SAMPLE_RATE, CORRECT_LABEL, INCORRECT_LABEL

logger = logging.getLogger(__name__)


def add_words(phrase, dict):
    for word in phrase.split(" "):
        dict[word] += 1


def clean_line(line):
    head, body = line.split("\t")

    body_set = set("".join(char for char in body if char not in string.punctuation).lower().split(" "))
    body_set.discard("")

    return "{}\t{}".format(head, " ".join(body_set))


def partition_dataset():
    os.remove(DATASET_FILE)
    correct_dict = defaultdict(int)
    incorrect_dict = defaultdict(int)

    with open(INPUT_FILE) as fin, open(DATASET_FILE, "a") as fout:
        for line in fin:
            line = clean_line(line)
            if random() < SAMPLE_RATE:
                process_sample_line(line, correct_dict, incorrect_dict)
            else:
                fout.write(line)

    return correct_dict, incorrect_dict


def process_sample_line(line, correct_dict, incorrect_dict):
    head, body = line.split("\n")
    if head == CORRECT_LABEL:
        add_words(body, correct_dict)
    elif head == INCORRECT_LABEL:
        add_words(body, incorrect_dict)
    else:
        raise KeyError("Key not recognised")
