"""
Name: training.py in Project: PytorchIntroduction
Author: Simon Leiner
Date: 28.05.23
Description: This file contains the batch training script
"""

import pandas as pd
import torch
from tqdm.auto import tqdm


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def training(EPOCHS: int, model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader,
             val_dataloader: torch.utils.data.DataLoader, loss_function: torch.nn.Module,
             optimizer: torch.optim.Optimizer, device: torch.device = "cpu"):
    """

    This function trains the model and prints the training and validation loss per epoch.

    :param EPOCHS: int: Number of epochs to train the model.
    :param model: object: Model to train.
    :param train_dataloader: object: Dataloader for the training data.
    :param val_dataloader: object: Dataloader for the validation data.
    :param loss_function: object: Loss function to use.
    :param optimizer: object: Optimizer to use.
    :param device: string: Device to use. Defaults to "cpu".
    :return: df_scores: pandas.DataFrame: Dataframe with the training and validation loss per epoch.
    """

    # empty lists to store the loss values
    train_loss_values = []
    val_loss_values = []
    epoch_count = []

    # create a training and test loop
    for epoch in tqdm(range(EPOCHS)):

        # printing
        print(f"Epoch: {epoch}\n-------")

        # train loss counter
        batch_train_loss = 0

        # model to train mode
        model.train()

        # training: loop thorugh the training batches
        for batch, (X_train, y_train) in enumerate(train_dataloader):

            # put data on device
            X_train, y_train = X_train.to(device), y_train.to(device)

            # calculate the forward pass
            y_pred_train = model(X_train)

            # calculate the training loss and add (accumulate) the loss to the counter
            training_loss = loss_function(y_pred_train, y_train)
            batch_train_loss += training_loss

            # optimizer zero grad
            optimizer.zero_grad()

            # calcuate the loss backwards (backpropagation)
            training_loss.backward()

            # optimizer step
            optimizer.step()

            # per batch printing every 1000
            if batch % 1000 == 0:
                print(f"Looked at {batch * len(X_train)} / {len(train_dataloader.dataset)} samples.")

        # divide total train loss by length of train dataloader: Average training loss per batch
        batch_train_loss /= len(train_dataloader)

        # validation loss counter
        batch_val_loss = 0

        # model to validation mode
        model.eval()

        # inference mode diasables gradient tracking
        with torch.inference_mode():

            # validation: loop thorugh the validation batches
            for batch, (X_val, y_val) in enumerate(val_dataloader):

                # put data on device
                X_val, y_val = X_val.to(device), y_val.to(device)

                # calculate the forward pass
                y_pred_val = model(X_val)

                # calculate the validation loss and add (accumulate) the loss to the counter
                val_loss = loss_function(y_pred_val, y_val)
                batch_val_loss += val_loss

            # divide total validation loss by length of val dataloader: Average validation loss per batch
            batch_val_loss /= len(val_dataloader)

        # append the loss values to the lists
        epoch_count.append(epoch)
        train_loss_values.append(training_loss.detach().numpy())
        val_loss_values.append(val_loss.detach().numpy())

        # print every 10 epochs
        if epoch % 10 == 0:

            # print
            print(f"Train Loss: {batch_train_loss:.5f} | Validation Loss:{batch_val_loss:.5f}\n")

    # convert the lists to pandas dataframe for plotting
    df_scores = pd.DataFrame({"Epoch": epoch_count, "Train Loss": train_loss_values, "Validation Loss": val_loss_values})

    return df_scores

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
