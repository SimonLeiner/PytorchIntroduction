"""
Module containing functions to make predictions on data.

- 'make_predictions': Makes a prediction on the data X with the model.

"""

import torch
import numpy as np


def make_predictions(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device = "cpu",
):
    """
    Makes a prediction on the data X with the model.

    Args:
        data_loader: torch.utils.data.DataLoader
            Data loader containing the data to make predictions on.
        model: torch.nn.Module
            Model to make predictions with.
        device: torch.device
            Device to make predictions on.
    Returns:
        pred_values: np.ndarray
            Predicted values.
        pred_values_prob: np.ndarray
            Predicted values as probabilities.
        true_values: np.ndarray
            True values.
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
            pred_values_prob.append(
                torch.amax(y_pred_prob, dim=1).detach().cpu().numpy()
            )
            true_values.append(y.detach().cpu().numpy())

    # flatten the lists
    pred_values = np.concatenate(pred_values).ravel()
    pred_values_prob = np.concatenate(pred_values_prob).ravel()
    true_values = np.concatenate(true_values).ravel()

    return pred_values, pred_values_prob, true_values
