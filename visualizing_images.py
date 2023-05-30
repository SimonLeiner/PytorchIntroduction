"""
Name: visualizing_images.py in Project: PytorchIntroduction
Author: Simon Leiner
Date: 30.05.23
Description: This file contains functions to visualize images
"""

import random
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def plot_random_images(image_path_list: list):
    """
    This function plots random images from the image_path.

    :param image_path_list: list: List of image paths.
    :return: None
    """

    # plot several images
    fig = plt.figure(figsize=(18, 9))
    rows, cols = 4, 8
    for i in range(1, rows * cols + 1):
        # get a random image path
        random_image_path = random.choice(image_path_list)

        # get image class from path name (the image class is the name of the directory where the image is stored)
        image_class = random_image_path.parent.stem

        # open and plot the image
        with Image.open(random_image_path) as img:
            plt.xlabel(f"Random Image width: {img.width}")
            plt.ylabel(f"Random Image height: {img.height}")
            fig.add_subplot(rows, cols, i)
            plt.imshow(img)
            plt.title(image_class)
            plt.axis(False)
    return None


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Checked: Function works

def plot_transformed_images(image_path_list: list, transform: transforms.Compose):
    """
    This function plots random images from the image_path.

    :param image_path_list: list: List of image paths.
    :param transform: transforms.Compose: Transformations to apply to the images.
    :return: None
    """

    # plot several images
    fig = plt.figure(figsize=(18, 9))
    rows, cols = 3, 8
    for i in range(1, rows * cols + 1, 2):
        # get a random image path
        random_image_path = random.choice(image_path_list)

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

    return None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
