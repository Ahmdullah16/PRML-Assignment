"""Architectures reused unchanged from the recovered PRML implementation.

Maintains checkpoint compatibility; logits are converted to softmax only at inference.
"""
import torch
from torch import nn
from . import LATENT


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(ni, no):
            return [nn.Conv2d(ni, no, 3, padding=1), nn.BatchNorm2d(no), nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*block(3, 16), *block(16, 32), *block(32, 64))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 64), nn.ReLU(),
                                  nn.Dropout(.3), nn.Linear(64, 4))

    def forward(self, x):
        return self.head(self.features(x))


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(4, 16)
        self.net = nn.Sequential(
            nn.Linear(LATENT + 16, 256 * 4 * 4), nn.ReLU(), nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh())

    def forward(self, z, y):
        return self.net(torch.cat((z, self.embed(y)), dim=1))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(.2), nn.Flatten())
        self.real_head = nn.Linear(256 * 4 * 4, 1)
        self.class_head = nn.Linear(256 * 4 * 4, 4)

    def forward(self, x):
        h = self.features(x)
        return self.real_head(h).flatten(), self.class_head(h)
