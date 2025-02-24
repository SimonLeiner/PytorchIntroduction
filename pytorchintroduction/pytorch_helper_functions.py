"""
Utility functions that are used in the PyTorch tutorials.

- 'create_writer': Creates a torch.utils.tensorboard.writer.SummaryWriter()
- 'walk_through_directory': Walks through a directory and prints the file names.
- 'set_global_seed': Sets the global seed for all random number generators.
- 'print_train_time': Prints the time it took to train the model.
"""

import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def create_writer(experiment_name: str, model_name: str, extra: str = None):
    """
    Creates a torch.utils.tensorboard.writer.SummaryWriter()
    instance saving to a specific log_dir, which
    is a combination of runs/timestamp/experiment_name/model_name/extra.
    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name: str
            Name of the experiment.
        model_name: str
            Name of the model.
        extra: str
            Extra string to add to the log_dir. Defaults to None.

    Returns:
        writer: torch.utils.tensorboard.writer.SummaryWriter
            SummaryWriter instance.
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


def walk_through_directory(dir_path: str):
    """
    Walks through a directory and prints the file names.

    Args:
        dir_path: str
            Path to the directory.

    Returns:
        None
    """
    # loop through the directory
    for dir_path, dir_names, file_names in os.walk(dir_path):
        print(
            f"There are {len(dir_names)} directories and {len(file_names)} images in '{dir_path}",
        )


def set_global_seed(seed: int = 42):
    """
    Sets the global seed for all random number generators.

    Args:
        seed: int
            Seed to set. Defaults to 42.

    Returns:
        None
    """
    # set the seed for numpy, torch and cuda
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def print_train_time(start: float, end: float, device: str = None):
    """
    Prints the time it took to train the model.

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
