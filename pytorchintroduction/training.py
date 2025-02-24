import pandas as pd
import torch
import torchmetrics
from tqdm.auto import tqdm


def training(  # noqa: PLR0913
    epochs: int,
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch_print: int,
    writer: torch.utils.tensorboard.writer.SummaryWriter,
    device: torch.device = "cpu",
) -> pd.DataFrame:
    """Train the model and print the training and validation loss per epoch.

    Args:
        epochs (int): Number of epochs to train the model.
        model (torch.nn.Module): PyTorch model to train.
        train_dataloader (torch.utils.data.DataLoader): PyTorch dataloader for training data.
        val_dataloader (torch.utils.data.DataLoader): PyTorch dataloader for validation data.
        loss_function (torch.nn.Module): PyTorch loss function.
        optimizer (torch.optim.Optimizer): PyTorch optimizer.
        epoch_print (int): Print the training and validation loss every epoch_print epochs.
        writer (torch.utils.tensorboard.writer.SummaryWriter): PyTorch SummaryWriter to log the training and validation loss.
        device (torch.device, optional): Device to run the model on. Defaults to "cpu".

    Returns:
        pd.DataFrame: DataFrame with the training and validation loss per epoch.
    """
    # print
    print("Starting training.")  # noqa: T201

    # dict with empty lists to store the loss values
    results = {
        "Epoch": [],
        "Train Loss": [],
        "Validation Loss": [],
        "Train Accuracy": [],
        "Validation Accuracy": [],
    }

    # create a training and test loop
    for epoch in tqdm(range(epochs)):
        # train loss counter
        batch_train_loss, batch_train_acc = 0, 0

        # model to train mode
        model.train()

        # training: loop thorugh the training batches
        for _, (X_train, y_train) in enumerate(train_dataloader):  # noqa: N806
            # put data on device
            X_train, y_train = X_train.to(device), y_train.to(device)  # noqa: N806, PLW2901

            # calculate the forward pass
            y_pred_train = model(X_train)

            # calculate the training loss and add (accumulate) the loss to the counter
            training_loss = loss_function(y_pred_train, y_train)
            batch_train_loss += training_loss
            batch_train_acc += torchmetrics.functional.accuracy(
                preds=y_pred_train.argmax(dim=1),
                target=y_train,
                task="multiclass",
                num_classes=y_pred_train.shape[1],
            )

            # optimizer zero grad
            optimizer.zero_grad()

            # calcuate the loss backwards (backpropagation)
            training_loss.backward()

            # optimizer step
            optimizer.step()

        # divide total train loss by length of train dataloader: Average training loss per batch
        batch_train_loss /= len(train_dataloader)
        batch_train_acc /= len(train_dataloader)

        # validation loss counter
        batch_val_loss, batch_val_acc = 0, 0

        # model to validation mode
        model.eval()

        # inference mode diasables gradient tracking
        with torch.inference_mode():
            # validation: loop thorugh the validation batches
            for _, (X_val, y_val) in enumerate(val_dataloader):  # noqa: N806
                # put data on device
                X_val, y_val = X_val.to(device), y_val.to(device)  # noqa: N806, PLW2901

                # calculate the forward pass
                y_pred_val = model(X_val)

                # calculate the validation loss and add (accumulate) the loss to the counter
                val_loss = loss_function(y_pred_val, y_val)
                batch_val_loss += val_loss
                batch_val_acc += torchmetrics.functional.accuracy(
                    preds=y_pred_val.argmax(dim=1),
                    target=y_val,
                    task="multiclass",
                    num_classes=y_pred_val.shape[1],
                )

            # divide total validation loss by length of val dataloader: Average validation loss per batch
            batch_val_loss /= len(val_dataloader)
            batch_val_acc /= len(val_dataloader)

        # append the loss values to the lists
        results["Epoch"].append(epoch)
        results["Train Loss"].append(batch_train_loss.detach().cpu().numpy())
        results["Validation Loss"].append(batch_val_loss.detach().cpu().numpy())
        results["Train Accuracy"].append(batch_train_acc.detach().cpu().numpy())
        results["Validation Accuracy"].append(batch_val_acc.detach().cpu().numpy())

        # Add loss results to SummaryWriter
        writer.add_scalars(
            main_tag="Loss",
            tag_scalar_dict={
                "Train Loss": batch_train_loss,
                "Validation Loss": batch_val_loss,
            },
            global_step=epoch,
        )

        # Add accuracy results to SummaryWriter
        writer.add_scalars(
            main_tag="Accuracy",
            tag_scalar_dict={
                "Train Accuracy": batch_train_acc,
                "Validation Accuracy": batch_val_acc,
            },
            global_step=epoch,
        )

        # Track the PyTorch model architecture, and pass in an example input
        writer.add_graph(
            model=model,
            input_to_model=torch.randn(32, 3, 224, 224).to(device),
        )

        # print every epochs
        if epoch % epoch_print == 0:
            # print
            print(f"Epoch: {epoch + 1} / {epochs}\n-------")  # noqa: T201
            print(  # noqa: T201
                f"Train Loss: {batch_train_loss:.5f} & Accuracy: {batch_train_acc:.5f} | Validation Loss:{batch_val_loss:.5f} & Accuracy: {batch_val_acc:.5f} \n",
            )

    # Close the writer
    writer.close()

    # convert the lists to pandas dataframe for plotting
    df_scores = pd.DataFrame(results)

    # print
    print("Finished training.")  # noqa: T201

    return df_scores
