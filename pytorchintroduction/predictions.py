import numpy as np
import torch


def make_predictions(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device = "cpu",
) -> tuple[np.array, np.array, np.array]:
    """Make a prediction on the data X with the model.

    Args:
        data_loader (torch.utils.data.DataLoader): PyTorch DataLoader with the data.
        model (torch.nn.Module): PyTorch model to make the prediction.
        device (torch.device, optional): Device to run the model on. Defaults to "cpu".

    Returns:
        np.array: Predicted values.
        np.array: Predicted values probabilities.
        np.array: True values.
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
        for _, (X, y) in enumerate(data_loader):  # noqa: N806
            # put data on device
            X, y = X.to(device), y.to(device)  # noqa: N806, PLW2901

            # Make a prediction
            y_pred = model(X)

            # convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
            y_pred_prob = torch.softmax(y_pred, dim=1)

            # convert prediction probabilities -> prediction labels (numerical)
            y_pred_final = torch.argmax(y_pred_prob, dim=1)

            # append
            pred_values.append(y_pred_final.detach().cpu().numpy())
            pred_values_prob.append(
                torch.amax(y_pred_prob, dim=1).detach().cpu().numpy(),
            )
            true_values.append(y.detach().cpu().numpy())

    # flatten the lists
    pred_values = np.concatenate(pred_values).ravel()
    pred_values_prob = np.concatenate(pred_values_prob).ravel()
    true_values = np.concatenate(true_values).ravel()

    return pred_values, pred_values_prob, true_values
