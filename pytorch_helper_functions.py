"""
Name: pytorch_helper_functions.py in Project: PytorchIntroduction
Author: Simon Leiner
Date: 27.05.23
Description: This file contains helper functions for the pytorch introduction
"""

import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works


def create_writer(experiment_name: str, model_name: str, extra: str = None):
    """
    This function creates a torch.utils.tensorboard.writer.SummaryWriter()
    instance saving to a specific log_dir, which
    is a combination of runs/timestamp/experiment_name/model_name/extra.
    Where timestamp is the current date in YYYY-MM-DD format.

    :param experiment_name: str: Name of experiment.
    :param model_name: str: Name of model.
    :param extra: str: Anything extra to add to the directory. Defaults to None.
    :return: torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.
    """

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    # print
    print(f"Created SummaryWriter, saving to: {log_dir}.")

    return SummaryWriter(log_dir=log_dir)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def walk_through_directory(dir_path: str):
    """
    This function walks through a directory and prints the file names.

    :param dir_path: str: Path to the directory.
    :return: None
    """

    # loop through the directory
    for dir_path, dir_names, file_names in os.walk(dir_path):
        print(f"There are {len(dir_names)} directories and {len(file_names)} images in '{dir_path}")

    return None


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def set_global_seed(seed: int = 42):
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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def print_train_time(start: float, end: float, device: str = None):
    """
    This function prints the time it took to train the model.

    :param start: float: Start time of computation (preferred in timeit format).
    :param end: float: End time of computation.
    :param device: str: Device that compute is running on. Defaults to None.
    :return: total_time: float: time between start and end in seconds (higher is longer).
    """

    # calculate the total time
    total_time = end - start

    # print the total time
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")

    return total_time


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def save_model(model: torch.nn.Module, path: str):
    """
    This function saves the model.

    :param model: object: Model to save.
    :param path: str: Path to save the model to.
    :return: None
    """

    # save the model
    torch.save(model.state_dict(), path)

    return None


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def load_model(model: torch.nn.Module, path: str):
    """
    This function loads the model.

    :param model: object: Model to load.
    :param path: str: Path to load the model from.
    :return: model: object: Loaded model.
    """

    # load the model
    model.load_state_dict(torch.load(path))

    return model

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
