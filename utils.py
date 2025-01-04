import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def one_hot_encoder_vector_from_labels(labels, n_classes):
    return F.one_hot(labels, num_classes=n_classes).float()

def concat_vectors(x, y):
    return torch.cat((x, y), dim=1)

def calculate_input_dim(z_dim, mnist_shape, n_classes):
    generator_input_dim = z_dim + n_classes
    discriminator_image_channel = mnist_shape[0] + n_classes
    return generator_input_dim, discriminator_image_channel




def plot_images_grid(real_images, fake_images, grid_size=5):
    """
    Plots real and fake images in a 5x5 grid.

    Args:
        real_images (torch.Tensor): Real images tensor.
        fake_images (torch.Tensor): Fake images tensor.
        grid_size (int, optional): Size of the grid (number of images per row/column). Default is 5.

    Returns:
        None. Displays a matplotlib plot.
    """
    real_images = (real_images + 1) / 2
    fake_images = (fake_images + 1) / 2

    real_images = real_images.detach().cpu()
    fake_images = fake_images.detach().cpu()

    fig, axes = plt.subplots(grid_size, 2 * grid_size, figsize=(10, 10))
    for i in range(grid_size):
        for j in range(grid_size):
            axes[i, j * 2].imshow(real_images[i * grid_size + j].squeeze(), cmap='gray')
            axes[i, j * 2].set_title("Real")
            axes[i, j * 2].axis("off")

            axes[i, j * 2 + 1].imshow(fake_images[i * grid_size + j].squeeze(), cmap='gray')
            axes[i, j * 2 + 1].set_title("Fake")
            axes[i, j * 2 + 1].axis("off")

    plt.tight_layout()
    plt.show()

def save_images_grid(real_images, fake_images, file_name, grid_size=5):
    """
    Saves real and fake images in a 5x5 grid to a file.

    Args:
        real_images (torch.Tensor): Real images tensor.
        fake_images (torch.Tensor): Fake images tensor.
        file_name (str): Path to save the image.
        grid_size (int, optional): Size of the grid (number of images per row/column). Default is 5.

    Returns:
        None. Saves the plot as an image file.
    """
    real_images = (real_images + 1) / 2
    fake_images = (fake_images + 1) / 2

    real_images = real_images.detach().cpu()
    fake_images = fake_images.detach().cpu()

    fig, axes = plt.subplots(grid_size, 2 * grid_size, figsize=(10, 10))
    for i in range(grid_size):
        for j in range(grid_size):
            axes[i, j * 2].imshow(real_images[i * grid_size + j].squeeze(), cmap='gray')
            axes[i, j * 2].set_title("Real")
            axes[i, j * 2].axis("off")

            axes[i, j * 2 + 1].imshow(fake_images[i * grid_size + j].squeeze(), cmap='gray')
            axes[i, j * 2 + 1].set_title("Fake")
            axes[i, j * 2 + 1].axis("off")

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close(fig)