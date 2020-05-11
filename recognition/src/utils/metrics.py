"""
Metric functions used during training and evaluation
"""

import editdistance

class BatchStringMetric:
    """
    Wraps any string metric function to be calculated on batch level
    """
    def __init__(self, metric_fn):
        self.metric_fn = metric_fn

    def __call__(self, first_batch, second_batch):
        assert len(first_batch) == len(second_batch)

        score = 0.0
        for first, second in zip(first_batch, second_batch):
            score += self.metric_fn(first, second)

        #print(first_batch[0])
        #print(second_batch[0])

        return score / len(first_batch)

@BatchStringMetric
def fragment_accuracy(first_str: str, second_str: str) -> float:
    """
    Calculate number of total string matches
    """
    return float(first_str == second_str)

@BatchStringMetric
def character_accuracy(first_str: str, second_str: str) -> float:
    """
    Use 1 - N.E.D from ICDAR 2019 RRC competitions
    https://rrc.cvc.uab.es/?ch=14&com=tasks
    1 - dist(s,t)/max(len(s), len(t))
    """
    if first_str == "" and second_str == "":
        return 1.0

    distance = editdistance.eval(first_str, second_str)
    norm_distance = distance / max(len(first_str), len(second_str))

    return 1.0 - norm_distance
