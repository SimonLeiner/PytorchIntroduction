"""
Name: make_predictions.py in Project: PytorchIntroduction
Author: Simon Leiner
Date: 28.05.23
Description: This file contains the prediction script
"""

import torch
import numpy as np


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def make_predictions(data_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device = "cpu"):
    """
    This function makes a prediction on the data X with the model.

    :param data_loader: torch.utils.data.DataLoader: Data to make a prediction on.
    :param model: object: Model to make a prediction with.
    :param device: string: Device to use. Defaults to "cpu".
    :return: pred_values, true_values: lists: Predictions and true value of the model.
    """

    pred_values = []
    pred_values_prob = []
    true_values = []

    # make sure the model is on the target device
    model.to(device)

    # turn on model evaluation mode
    model.eval()

    # turn on inference mode
    with torch.inference_mode():
        # training: loop thorugh the training batches
        for batch, (X, y) in enumerate(data_loader):

            # put data on device
            X, y = X.to(device), y.to(device)

            # Make a prediction
            y_pred = model(X)

            # convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
            y_pred_prob = torch.softmax(y_pred, dim=1)

            # convert prediction probabilities -> prediction labels (numerical)
            y_pred_final = torch.argmax(y_pred_prob, dim=1)

            # append
            pred_values.append(y_pred_final.detach().cpu().numpy())
            pred_values_prob.append(torch.amax(y_pred_prob, dim=1).detach().cpu().numpy())
            true_values.append(y.detach().cpu().numpy())

    # flatten the lists
    pred_values = np.concatenate(pred_values).ravel()
    pred_values_prob = np.concatenate(pred_values_prob).ravel()
    true_values = np.concatenate(true_values).ravel()

    return pred_values, pred_values_prob, true_values

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
