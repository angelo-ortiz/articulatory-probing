import functools
import operator
import random

import numpy as np
import torch


def flatten_lists(lists_2d):
    return functools.reduce(operator.iconcat, lists_2d, [])


def flatten_tuples(tuples_2d):
    return functools.reduce(operator.iconcat, tuples_2d, ())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
