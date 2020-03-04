import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(730, 729)
        self.conv1 = nn.Conv2d(9, 32, 5)   # >> 5x5
        self.conv2 = nn.Conv2d(32, 64, 3)  # >> 3x3
        self.conv3 = nn.Conv2d(64, 81, 3)  # >> 1x1

        def forward(self, x):
            x = self.fc1(x)
            x = F.leaky_relu(x)
            x = x.view(9, 9, 9)
            x = self.conv1(x)
            x = F.leaky_relu(x)
            x = self.conv2(x)
            x = F.leaky_relu(x)
            x = self.conv3(x)
            x = F.softmax(x)
            output = x.view(9, 9)
            return output
