import logging
from enum import Enum, unique
from math import log

from ex_2.evaluation import evaluate

logger = logging.getLogger(__name__)


@unique
class Algorithm(Enum):
    BERNOULLI = "bernoulli"
    MULTINOMIAL = "multinomial"

    @classmethod
    def learn_bernoulli(cls, line, corrects, incorrects, correct_dict, incorrect_dict):
        head, body = line.split("\t")
        correct_prob = corrects / (corrects + incorrects)
        incorrect_prob = incorrects / (corrects + incorrects)
        found = False

        for word in body.split(" "):
            if word in correct_dict and word not in incorrect_dict:
                logger.info("{}\n{}\n".format(body, evaluate(head, 1, 0)))
                found = True
                break
            if word not in correct_dict and word in incorrect_dict:
                found = True
                logger.info("{}\n{}\n".format(body, evaluate(head, 0, 1)))
                break
            if word in correct_dict and word in incorrect_dict:
                correct_prob += log(correct_dict[word]/(corrects + incorrects))
                incorrect_prob += log(incorrect_dict[word]/(corrects + incorrects))

        if not found:
            logger.info("{}\n{}\n".format(body, evaluate(head, correct_prob, incorrect_prob)))

    @classmethod
    def learn_multinomial(cls, line, corrects, incorrects, correct_dict, incorrect_dict, total_words, unique_words):
        from ex_2 import LAPLACE_SMOOTHING
        head, body = line.strip("\n").split("\t")
        correct_prob = corrects / (corrects + incorrects)
        incorrect_prob = incorrects / (corrects + incorrects)
        found = False

        if LAPLACE_SMOOTHING:
            for word in body.split(" "):
                correct_prob += log((correct_dict[word] + 1)/(total_words + len(unique_words)))
                incorrect_prob += log((incorrect_dict[word] + 1)/(total_words + len(unique_words)))
        else:
            for word in body.split(" "):
                if word in correct_dict and word not in incorrect_dict:
                    logger.info("{}\n{}\n".format(body, evaluate(head, 1, 0)))
                    found = True
                    break
                if word not in correct_dict and word in incorrect_dict:
                    found = True
                    logger.info("{}\n{}\n".format(body, evaluate(head, 0, 1)))
                    break
                if word in correct_dict and word in incorrect_dict:
                    correct_prob += log(correct_dict[word]/total_words)
                    incorrect_prob += log(incorrect_dict[word]/total_words)

        if not found:
            logger.info("{}\n{}\n".format(body, evaluate(head, correct_prob, incorrect_prob)))

    def delete_duplicates(self):
        if self is Algorithm.BERNOULLI:
            return True
        elif self is Algorithm.MULTINOMIAL:
            return False
        else:
            raise KeyError("Unknown algorithm")

    def learn(self, corrects, incorrects, correct_dict, incorrect_dict):
        from ex_2 import DATASET

        if self is Algorithm.MULTINOMIAL:
            unique_words = set(correct_dict).union(set(incorrect_dict))
            total_words = sum(correct_dict.values()) + sum(incorrect_dict.values())

        with open(DATASET) as fin:
            for line in fin:
                line = line.strip("\n")
                if self is Algorithm.BERNOULLI:
                    Algorithm.learn_bernoulli(line, corrects, incorrects, correct_dict, incorrect_dict)
                elif self is Algorithm.MULTINOMIAL:
                    Algorithm.learn_multinomial(line, corrects, incorrects, correct_dict, incorrect_dict,
                                                total_words, unique_words)
                else:
                    raise KeyError("Unknown algorithm")
