import numpy as np
from sklearn.metrics.cluster import rand_score
from sklearn.metrics import jaccard_score


def rand_index(true_fore, inferred_fore):
    return rand_score(true_fore.flatten(), inferred_fore.flatten())


def segmentation_accuracy(true_fore, inferred_fore):
    return jaccard_score(true_fore.flatten(), inferred_fore.flatten())
