import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(730, 729)
        self.conv1 = nn.Conv2d(9, 32, 5)   # >> 5x5
        self.conv2 = nn.Conv2d(32, 64, 3)  # >> 3x3
        self.conv3 = nn.Conv2d(64, 1, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = x.view(1, 9, 9, 9)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        output = x.flatten()
        return output
