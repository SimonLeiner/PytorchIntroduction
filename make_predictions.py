"""
Name: make_predictions.py in Project: PytorchIntroduction
Author: Simon Leiner
Date: 28.05.23
Description: This file contains the prediction script
"""

import torch


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def make_predictions(X: torch.Tensor, model: torch.nn.Module, device: torch.device = "cpu"):
    """
    This function makes a prediction on the data X with the model.

    :param X: torch.Tensor: Data to make a prediction on.
    :param model: object: Model to make a prediction with.
    :param device: string: Device to use. Defaults to "cpu".
    :return: predictions: torch.Tensor: Predictions of the model.
    """

    # make sure the model is on the target device
    model.to(device)

    # turn on model evaluation mode
    model.eval()

    # turn on inference mode
    with torch.inference_mode():
        # make sure the data is on the target device
        X = X.to(device)

        # Make a prediction
        predictions = model(X)

    return predictions

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
