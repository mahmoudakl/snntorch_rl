import torch

import numpy as np


def two_neuron_encoding(x):
    """
    convert an input vector to a two-neuron representation

    :param x: mixed-sign input vector
    """
    y = []
    for i in x:
        if i >= 0:
            y.append(i)
            y.append(0)
        else:
            y.append(0)
            y.append(abs(i))

    return torch.tensor(y)
