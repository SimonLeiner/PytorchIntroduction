import secrets

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision import transforms


def plot_random_images(image_path_list: list) -> None:
    """Plot random images from the image_path.

    Args:
        image_path_list (list): List of image paths.
    """
    # plot several images
    fig = plt.figure(figsize=(18, 9))
    rows, cols = 4, 8
    for i in range(1, rows * cols + 1):
        # get a random image path
        random_image_path = secrets.randbelow(image_path_list)

        # get image class from path name (the image class is the name of the directory where the image is stored)
        image_class = random_image_path.parent.stem

        # open and plot the image
        with Image.open(random_image_path) as img:
            plt.xlabel(f"Random Image width: {img.width}")
            plt.ylabel(f"Random Image height: {img.height}")
            fig.add_subplot(rows, cols, i)
            plt.imshow(img)
            plt.title(image_class)


def plot_transformed_images(
    image_path_list: list,
    transform: transforms.Compose,
) -> None:
    """Plot transformed random images from the image_path.

    Args:
        image_path_list (list): List of image paths.
        transform (transforms.Compose): Transformations to apply to the images.
    """
    # plot several images
    fig = plt.figure(figsize=(18, 9))
    rows, cols = 3, 8
    for i in range(1, rows * cols + 1, 2):
        # get a random image path
        random_image_path = secrets.randbelow(image_path_list)

        # get image class from path name (the image class is the name of the directory where the image is stored)
        image_class = random_image_path.parent.stem

        # open and plot the images
        with Image.open(random_image_path) as img:
            fig.add_subplot(rows, cols, i)
            plt.imshow(img)
            plt.title(f"Original ({image_class}) \nSize: {img.size}")
            plt.axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(img).permute(1, 2, 0)
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(transformed_image)
            plt.title(f"Transformed \nSize: {transformed_image.shape}")
            plt.axis("off")


def plot_confusionmatrix(  # noqa: PLR0913
    y_pred_train: pd.Series,
    y_true_train: pd.Series,
    y_pred_test: pd.Series,
    y_true_test: pd.Series,
    class_names: list,
    normalize: str | None = None,
) -> None:
    """Plot the confusion matrix.

    Args:
        y_pred_train (pd.Series): Predicted values for the training set.
        y_true_train (pd.Series): True values for the training set.
        y_pred_val (pd.Series): Predicted values for the validation set.
        y_true_val (pd.Series): True values for the validation set.
        class_names (list): List of class names.
        normalize (str, optional): Normalize the confusion matrix. Defaults to None.
    """
    # get the confusion matrix
    mat_train = confusion_matrix(y_true_train, y_pred_train, normalize=normalize)
    mat_val = confusion_matrix(y_true_test, y_pred_test, normalize=normalize)

    # create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle("Confusion Matrix: \n")
    sns.heatmap(
        mat_train,
        cbar=True,
        annot=True,
        square=True,
        fmt=".2f",
        annot_kws={"size": 10},
        ax=ax1,
        xticklabels=class_names,
        yticklabels=class_names,
    )

    ax1.set_title("Training Confusion Matrix")
    ax1.set_xlabel("Predicted label:")
    ax1.set_ylabel("True label:")

    sns.heatmap(
        mat_val,
        cbar=True,
        annot=True,
        square=True,
        fmt=".2f",
        annot_kws={"size": 10},
        ax=ax2,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax2.set_title("Testing Confusion Matrix")
    ax2.set_xlabel("Predicted label:")
    ax2.set_ylabel("True label:")


def plot_loss_curve(df_scores: pd.DataFrame) -> None:
    """Plot the loss and accuracy curves.

    Args:
        df_scores (pd.DataFrame): DataFrame with the training and validation loss per epoch.
    """
    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(df_scores["Epoch"], df_scores["Train Loss"], label="Train Loss")
    plt.plot(df_scores["Epoch"], df_scores["Validation Loss"], label="Validation Loss")
    plt.title("Loss:")
    plt.xlabel("Epochs:")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(df_scores["Epoch"], df_scores["Train Accuracy"], label="Train Accuracy")
    plt.plot(
        df_scores["Epoch"],
        df_scores["Validation Accuracy"],
        label="Validation Accuracy",
    )
    plt.title("Accuracy:")
    plt.xlabel("Epochs:")
    plt.legend()
