
from os import path

import torch
import torch.nn.functional as F
import torch.optim as optim

from mcts import MCTS
from policy_net import PolicyNet
from value_net import ValueNet


class ActorCritic:
    def __init__(self):
        self.mcts = None
        self.policy_net = PolicyNet()
        self.value_net = ValueNet()
        self.optim = optim.SGD(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=1e-2, momentum=0.9)

    def new_game(self):
        self.mcts = MCTS()

    def step(self):
        self.mcts.step(self.policy_net, self.value_net)

    def select_action(self):
        return self.mcts.select_most()

    def place(self, action):
        return self.mcts.place(self.policy_net, action)

    def train(self, end_state, histories):
        policy_losses = []
        value_losses = []

        def get_reward(state):
            return float(end_state.reward if end_state.turn == state.turn else -end_state.reward)

        for state, probs_target in histories:
            probs = self.policy_net.forward(state.state)
            probs_target = torch.tensor(probs_target)

            policy_losses.append(-torch.sum(probs_target * torch.log(probs)))

            value = self.value_net.forward(state.state)
            value_target = torch.tensor(get_reward(state)).unsqueeze(0)

            value_losses.append(F.mse_loss(value, value_target))

            print(value)
            print(value_target)

        loss = torch.sum(torch.stack(policy_losses) +
                         torch.stack(value_losses))
        print('loss:', loss.item())

        self.optim.zero_grad()

        loss.backward()

        self.optim.step()

    def save(self, name):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
        }, path.join(path.dirname(path.dirname(__file__)), 'saves', name))

    def load(self, name):
        checkpoint = torch.load(
            path.join(path.dirname(path.dirname(__file__)), 'saves', name))

        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])
