import torch
import torch.nn as nn
import torch.nn.functional as F


# class PolicyNet(nn.Module):
#     def __init__(self):
#         super(PolicyNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 8, 3, padding=1)     # >> 3x3
#         self.conv2 = nn.Conv2d(8, 16, 3, padding=1)    # >> 3x3
#         self.conv3 = nn.Conv2d(16, 32, 3, padding=1)   # >> 3x3
#         self.conv4 = nn.Conv2d(32, 9, 3)               # >> 1x1

#     def forward(self, x):
#         x = x.view(1, 3, 3, 3)
#         x = self.conv1(x)
#         x = F.leaky_relu(x)
#         x = self.conv2(x)
#         x = F.leaky_relu(x)
#         x = self.conv3(x)
#         x = F.leaky_relu(x)
#         x = self.conv4(x)
#         x = x.flatten()
#         x = F.softmax(x, -1)
#         output = x
#         return output

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(27, 36)
        self.fc2 = nn.Linear(36, 36)
        self.policy_head = nn.Linear(36, 9)
        self.value_head = nn.Linear(36, 1)

    def forward(self, x):
        x = x.view(1, 27)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        policy = self.policy_head(x)
        value = self.value_head(x)

        policy = policy.flatten()
        value = value.flatten()

        policy = F.softmax(policy, 0)
        value = torch.tanh(value)

        return policy, value
