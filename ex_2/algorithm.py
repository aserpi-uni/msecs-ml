from enum import Enum, unique


@unique
class AlgorithmType(Enum):
    BERNOULLI = "bernoulli"
    MULTINOMIAL = "multinomial"

    def delete_duplicates(self):
        if self is AlgorithmType.BERNOULLI:
            return True
        elif self is AlgorithmType.MULTINOMIAL:
            return False
        else:
            raise KeyError("Unknown algorithm")
