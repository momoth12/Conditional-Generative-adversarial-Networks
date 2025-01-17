import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, image_channel=1):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.gen = nn.Sequential(
            self.generator_block(input_dim, hidden_dim * 4),
            self.generator_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.generator_block(hidden_dim * 2, hidden_dim),
            self.generator_block(hidden_dim, image_channel, kernel_size=4, final_layer=True),
        )

    def generator_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        return self.gen(noise.view(len(noise), self.input_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, image_channel, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.discriminator_block(image_channel, hidden_dim),
            self.discriminator_block(hidden_dim, hidden_dim * 2),
            self.discriminator_block(hidden_dim * 2, 1, final_layer=True),
        )

    def discriminator_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size, stride))

    def forward(self, image):
        pred = self.disc(image)
        return pred.view(len(pred), -1)
