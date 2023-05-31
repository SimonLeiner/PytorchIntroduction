"""
Name: validation.py in Project: PytorchIntroduction
Author: Simon Leiner
Date: 28.05.23
Description:
"""

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def get_confusionmatrix(y_pred_train: pd.Series, y_true_train: pd.Series,
                        y_pred_val: pd.Series, y_true_val: pd.Series,
                        class_names: list, normalize: str = None):
    """

    This function plots the confusion matrix. SEE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

    :param y_pred_train: pd.Series: predictions py the build_model
    :param y_true_train: pd.Series: true y values
    :param y_pred_val: pd.Series: predictions py the build_model
    :param y_true_val: pd.Series: true y values
    :param class_names: list: list of class names
    :param normalize: boolean: 'true' => normalizes the matrix
    :return: None
    """

    # get the confusion matrix
    mat_train = confusion_matrix(y_true_train, y_pred_train, normalize=normalize)
    mat_val = confusion_matrix(y_true_val, y_pred_val, normalize=normalize)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    fig.suptitle('Confusion Matrix: \n')

    sns.heatmap(
        mat_train,
        cbar=True,
        annot=True,
        square=True,
        fmt=".2f",
        annot_kws={"size": 10},
        ax=ax1,
        xticklabels=class_names,
        yticklabels=class_names
    )

    ax1.set_title('Training Confusion Matrix')
    ax1.set_xlabel('Predicted label:')
    ax1.set_ylabel('True label:')

    sns.heatmap(
        mat_val,
        cbar=True,
        annot=True,
        square=True,
        fmt=".2f",
        annot_kws={"size": 10},
        ax=ax2,
        xticklabels=class_names,
        yticklabels=class_names
    )
    ax2.set_title('Validation Confusion Matrix')
    ax2.set_xlabel('Predicted label:')
    ax2.set_ylabel('True label:')

    return None


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def plot_loss_curve(df_scores: pd.DataFrame):
    """
    This function plots the loss and accuracy curves.

    :param df_scores: pd.DataFrame: Dataframe with the scores
    :return: None
    """

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(df_scores["Epoch"], df_scores["Train Loss"], label='Train Loss')
    plt.plot(df_scores["Epoch"], df_scores["Validation Loss"], label='Validation Loss')
    plt.title('Loss:')
    plt.xlabel('Epochs:')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(df_scores["Epoch"], df_scores["Train Accuracy"], label='Train Accuracy')
    plt.plot(df_scores["Epoch"], df_scores["Validation Accuracy"], label='Validation Accuracy')
    plt.title('Accuracy:')
    plt.xlabel('Epochs:')
    plt.legend()

    return None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
