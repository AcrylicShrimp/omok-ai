
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
        self.policy_optim = optim.Adam(self.policy_net.parameters())
        self.value_optim = optim.Adam(self.value_net.parameters())

    def new_game(self):
        self.mcts = MCTS()

    def step(self, side):
        self.mcts.step(side, self.policy_net, self.value_net)

    def select_action(self):
        return self.mcts.select_most()

    def place(self,  action):
        self.mcts.place(self.policy_net, action)

    def train(self, end_state):
        policy_losses = []
        value_losses = []

        def get_reward(state):
            return end_state.reward if state.turn == end_state.turn else -end_state.reward

        for history in self.mcts.histories:
            value_losses.append(F.mse_loss(
                self.value_net.forward(history[0].state), torch.Tensor([get_reward(history[0])])).unsqueeze(0))

            print(self.policy_net.forward(history[0].state).size())
            print(history[1].size())

            policy_losses.append(F.cross_entropy(
                self.policy_net.forward(history[0].state), history[1]))

        loss = torch.sum(torch.stack(policy_losses) +
                         torch.stack(value_losses))

        self.policy_optim.zero_grad()
        self.value_optim.zero_grad()

        loss.backward()

        self.policy_optim.step()
        self.value_optim.step()

    def save(self, name):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optim_state_dict': self.policy_optim.state_dict(),
            'value_optim_state_dict': self.value_optim.state_dict(),
        }, path.join(path.dirname(path.dirname(__file__)), 'saves', name))

    def load(self, name):
        checkpoint = torch.load(
            path.join(path.dirname(path.dirname(__file__)), 'saves', name))

        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optim.load_state_dict(
            checkpoint['policy_optim_state_dict'])
        self.value_optim.load_state_dict(checkpoint['value_optim_state_dict'])
