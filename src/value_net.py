import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)    # >> 7x7
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)   # >> 5x5
        self.bnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3)  # >> 3x3
        self.bnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 1, 3)   # >> 1x1

    def forward(self, x):
        x = x.view(1, 3, 9, 9)
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.bnorm2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = self.bnorm3(x)
        x = F.leaky_relu(x)
        x = self.conv4(x)
        x = torch.tanh(x)
        output = x.flatten()
        return output
