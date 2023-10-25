import numpy as np
import torch


def fix_seeds(seed):
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_weights_kaiming_normal(module):
    if isinstance(module, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity='leaky_relu')

