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

def get_confusionmatrix(y_true: pd.Series, y_pred: pd.Series, normalize: str = None):
    """

    This function plots the confusion matrix. SEE: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

    :param y_true: pd.Series: true y values
    :param y_pred: pd.Series: predictions py the build_model
    :param normalize: boolean: 'true' => normalizes the matrix
    :return: None
    """

    # get the confusion matrix
    mat = confusion_matrix(y_true, y_pred, normalize=normalize)

    # extract values:
    tn, fp, fn, tp = mat.ravel()
    pos = tp + fp
    neg = tn + fn
    total = tn + fp + fn + tp

    print(f"There are a Total of {total} predictions:")
    print(f"The build_model predicted {pos} times ({round((pos / total), 2)}) % ture positives.")
    print(f"The build_model predicted {neg} times ({round((neg / total), 2)}) % true negatives.")
    print(f"The build_model predicted in total {round(((tp + tn) / total), 2)} % of the predictions correctly.")
    print("-" * 10)

    # view it as heatmap:
    sns.heatmap(
        mat,
        cbar=True,
        annot=True,
        square=True,
        fmt=".2f",
        annot_kws={"size": 10},
    )

    # set stylisitcal stuff
    plt.title('Confusion Matrix: \n')
    plt.xlabel("Predicted label:")
    plt.ylabel("True label:")
    plt.show()

    return None


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def plot_loss_curve(df_scores: pd.DataFrame):
    """"""

    # plot the loss curve
    sns.lineplot(data=pd.melt(df_scores, ['Epoch'], x="Epoch", y="value", hue="variable"))

    # set stylisitcal stuff
    plt.title('Loss Curve: \n')
    plt.xlabel("Epochs:")
    plt.ylabel("Loss:")
    plt.show()

    return None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
