from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        # convolutional layers
        self._body = nn.Sequential(
            # First convolutional layer
            # input size = (32, 32), output size =(28, 28)
            # out = ⌊(in + 2*padding − dilation*(kernel−1) − 1)/stride⌋ + 1
            nn.Conv2d(
                in_channels=1,  # wrapped in 4D (N, C_in, H, W)
                out_channels=6,
                kernel_size=5),  # 32 -> 28
            # Relu activation
            nn.ReLU(inplace=True),
            # Max pool 2-d
            nn.MaxPool2d(kernel_size=2),  # 28 -> 14

            # Second convolutional layer
            # input size = (14, 14), output size = (10, 10)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),  # 14->10
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 10->5
            # output size = (5, 5)
        )

        self._head = nn.Sequential(
            # First fully connected layer
            # in_features = total numbers of weights in last conv layer = 16 * 5 * 5
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.ReLU(inplace=True),

            # Second fully connected layer
            # in_features = output of last linear layer = 84
            # and out features = number of classes = 10 (MNIST data 0-9)
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(inplace=True),

            # Third fully connected layer. It is also output layer
            # in_features = output of last linear layer = 84
            # and out_features = number of classes = 10 (MNIST data 0-9)
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply feature extractor
        x = self._body(x)

        # flatten the output conv layers
        # dimension should be batch_size * number_of_weights_in_last_conv_layer
        x = x.view(x.size()[0], -1)

        # apply classification head
        x = self._head(x)
        return x

    def get_data(self, batch_size: int, data_root: 'data', num_workers=1) -> tuple[DataLoader[Any], DataLoader[Any]]:
        train_test_transforms = transforms.Compose([
            # resize to 32x32
            transforms.Resize((32, 32)),
            # this rescales image tensor values between 0-1. image_tensor /= 255
            transforms.ToTensor(),
            # subtract mean (0.1307) and divide by variance (0.3081).
            # This mean and variance is calculated on training data (verify yourself)
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_loader = DataLoader(
            datasets.MNIST(root=data_root,
                           train=True,
                           download=True,
                           transform=train_test_transforms),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        test_loader = DataLoader(
            datasets.MNIST(root=data_root,
                           train=False,
                           download=True,
                           transform=train_test_transforms),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, test_loader
