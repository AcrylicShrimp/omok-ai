

import math
import random

import torch
from torch.distributions.dirichlet import Dirichlet

from game import Game


class MCTSNode:
    def __init__(self, state, p=1., parent=None, action=None):
        self.state = state
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = p
        self.childs = {}
        self.parent = parent
        self.action = action

    @property
    def is_leaf(self):
        return len(self.childs) == 0

    def calc_ucb(self, c_puct):
        return self.q + c_puct * self.p * (math.sqrt(self.parent.n) if self.parent is not None else 0) / (1 + self.n)

    def select_most(self):
        if self.is_leaf:
            return None

        # Gets childs.
        childs = [(child.n, child)
                  for action, child in self.childs.items()]

        # Shuffle them - to randomly select a child when there're many childs that have same visit count.
        random.shuffle(childs)

        # Sort them and returns first child.
        return sorted(childs, key=lambda x: x[0], reverse=True)[0][1]

    def select_leaf(self):
        if self.is_leaf:
            return self

        # Gets childs.
        childs = [(child.calc_ucb(1.), child)
                  for action, child in self.childs.items()]

        # Shuffle them - to randomly select a child when there're many childs that have same UCB value.
        random.shuffle(childs)

        # Sort them and returns recursive call.
        return sorted(childs, key=lambda x: x[0], reverse=True)[0][1].select_leaf()

    def expand(self, policy, noise):
        if not self.is_leaf:
            return

        if self.state.is_terminated:
            return

        probs = policy.forward(self.state.state)
        actions = Game.possible_actions(self.state)

        dirichlet = Dirichlet(torch.ones(len(actions))
                              * 0.3).sample([1]).flatten()

        # Gets all possible actions and creates child nodes for each action.
        for index, action in enumerate(actions):
            prob = probs[action].item()

            if noise:
                prob = 0.75 * prob + 0.25 * dirichlet[index].item()

            self.childs[action] = MCTSNode(
                Game.next_state(self.state, action), prob, self, action)

    def backup(self, policy, value):
        reward = self.state.reward \
            if self.state.is_terminated \
            else value.forward(self.state.state).item()

        node = self

        while node is not None:
            node.n += 1
            node.w += -reward * (1 if self.state.turn ==
                                 node.state.turn else -1)
            node.q = node.w / node.n
            node = node.parent


class MCTS:
    def __init__(self):
        self.noise = True
        self.root = MCTSNode(Game.init())

    def step(self, policy, value):
        leaf = self.root.select_leaf()
        leaf.expand(policy, self.noise)
        leaf = leaf.select_leaf()
        leaf.backup(policy, value)

        # Turn off dirichlet noise.
        self.noise = False

    def select_most(self):
        return self.root.select_most().action

    def place(self, policy, action):
        if action not in self.root.childs:
            self.root.expand(policy, False)

            if action not in self.root.childs:
                raise RuntimeError('given action is not legal')

        probs_target = [self.root.childs[action].n /
                        self.root.n if action in self.root.childs else 0 for action in Game.all_actions()]
        history = (self.root.state, probs_target)

        self.root = self.root.childs[action]
        self.root.parent = None
        self.root.action = None

        return history
