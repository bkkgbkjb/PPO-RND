from torch import nn
import torch
import numpy as np


def init_module_weights(layer):
    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
    nn.init.constant_(layer.bias, 0.0)
    return layer
