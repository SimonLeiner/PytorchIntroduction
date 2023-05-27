"""
Name: pytorch_helper_functions.py in Project: PytorchIntroduction
Author: Simon Leiner
Date: 27.05.23
Description: This file contains helper functions for the pytorch introduction
"""

import numpy as np
import torch


def set_global_seed(seed=42):
    """
    This function sets the global seed for all random number generators.

    :param seed: int: seed to set
    :return: None
    """

    # set the seed for numpy, torch and cuda
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    return None
