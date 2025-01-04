import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import nn

class MNISTDataModule(nn.Module):
    def __init__(self, batch_size):
        super(MNISTDataModule, self).__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.mnist_train = MNIST(root='data', train=True, download=True, transform=self.transform)
        self.mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)
        self.mnist_val = MNIST(root='data', train=False, download=True, transform=self.transform)
        self.mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=False)

    def train_dataloader(self):
        return self.mnist_train

    def val_dataloader(self):
        return self.mnist_val

    def test_dataloader(self):
        return self.mnist_val
