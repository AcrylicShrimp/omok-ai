
from os import path

import torch
import torch.nn.functional as F
import torch.optim as optim

from mcts import MCTS
from network import Network


class ActorCritic:
    def __init__(self):
        self.mcts = None
        self.network = Network()
        self.optim = optim.Adam(self.network.parameters())

    def new_game(self):
        self.mcts = MCTS()

    def step(self):
        self.mcts.step(self.network)

    def select_action(self):
        return self.mcts.select_best()

    def place(self, action, train=False):
        return self.mcts.place(action, train)

    def train(self, end_state, histories):
        policy_losses = []
        value_losses = []

        def get_reward(state):
            return float(end_state.reward if end_state.turn == state.turn else -end_state.reward)

        for state, probs_target in histories:
            probs, value = self.network.forward(state.state)

            probs_target = torch.tensor(probs_target)
            value_target = torch.tensor(get_reward(state)).unsqueeze(0)

            policy_losses.append(-torch.sum(probs_target * torch.log(probs)))
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
            'network_state_dict': self.network.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
        }, path.join(path.dirname(path.dirname(__file__)), 'saves', name))

    def load(self, name):
        checkpoint = torch.load(
            path.join(path.dirname(path.dirname(__file__)), 'saves', name))

        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])
