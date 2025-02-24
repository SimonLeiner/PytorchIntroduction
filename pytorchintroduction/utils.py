import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def create_writer(
    experiment_name: str,
    model_name: str,
    extra: str | None = None,
) -> SummaryWriter:
    """Create a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir, which is a combination of runs/timestamp/experiment_name/model_name/extra. Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of the experiment.
        model_name (str): Name of the model.
        extra (str, optional): Extra information to add to the log directory. Defaults to None.

    Returns:
        SummaryWriter: SummaryWriter

    """
    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d")

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)  # noqa: PTH118
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)  # noqa: PTH118

    # print
    print(f"Created SummaryWriter, saving to: {log_dir}.")  # noqa: T201

    return SummaryWriter(log_dir=log_dir)


def walk_through_directory(dir_path: str) -> None:
    """Walk through a directory and prints the file names.

    Args:
        dir_path (str): Path to the directory.
    """
    # loop through the directory
    for path, dir_names, file_names in os.walk(dir_path):
        print(  # noqa: T201
            f"There are {len(dir_names)} directories and {len(file_names)} images in '{path}",
        )


def set_global_seed(seed: int = 42) -> None:
    """Set the global seed for all random number generators.

    Args:
        seed (int, optional): Seed to set. Defaults to 42.
    """
    # set the seed for numpy, torch and cuda
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train_time(start: float, end: float, device: str | None = None) -> float:
    """Print the time it took to train the model.

    Args:
        start (float): Start time.
        end (float): End time.
        device (str, optional): Device used for training. Defaults to None.

    Returns:
        total_time: float
            Total time it took to train the model.
    """
    # calculate the total time
    total_time = end - start

    # print the total time
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")  # noqa: T201

    return total_time
