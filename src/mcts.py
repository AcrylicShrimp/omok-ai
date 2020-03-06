

import math
import random

import torch

from game import Game


class MCTSNode:
    def __init__(self, state, ubc=0, parent=None, action=None):
        self.state = state
        self.ucb = ubc
        self.reward = 0
        self.visited = 0
        self.childs = {}
        self.parent = parent
        self.action = action

    @property
    def is_leaf(self):
        return len(self.childs) == 0

    def select_most(self):
        if self.is_leaf:
            return None

        # Gets childs.
        childs = [(child.visited, child)
                  for action, child in self.childs.items()]

        # Shuffle them - to randomly select a child when there're many childs that have same visit count.
        random.shuffle(childs)

        # Sort them and returns first child.
        return sorted(childs, key=lambda x: x[0], reverse=True)[0][1]

    def select_leaf(self):
        if self.is_leaf:
            return self

        # Gets childs.
        childs = [(child.ucb, child)
                  for action, child in self.childs.items()]

        # Shuffle them - to randomly select a child when there're many childs that have same UCB value.
        random.shuffle(childs)

        # Sort them and returns recursive call.
        return sorted(childs, key=lambda x: x[0], reverse=True)[0][1].select_leaf()

    def expand(self, policy):
        if not self.is_leaf:
            return

        if self.state.is_terminated:
            return

        child_prob = policy.forward(self.state.state)
        visited_sqrt = math.sqrt(self.visited)

        # Gets all possible actions and creates child nodes for each action.
        for action in Game.possible_actions(self.state):
            next_state = Game.next_state(self.state, action)
            child_ucb = child_prob[action] * visited_sqrt
            self.childs[action] = MCTSNode(next_state, child_ucb, self, action)

    def backup(self, side, policy, value):
        reward = self.state.reward if self.state.is_terminated else value.forward(
            self.state.state)

        if side != self.state.turn:
            reward = -reward

        node = self

        while node is not None:
            node.visited += 1
            node.reward += reward * (1 if side == node.state.turn else -1)

            if self.parent is not None and self.action is not None:
                node.ucb = (node.reward + policy.forward(self.parent.state.state)[self.action] * math.sqrt(
                    self.parent.visited)) / (1 + self.visited)

            node = node.parent


class MCTS:
    def __init__(self):
        self.root = MCTSNode(Game.init())
        self.histories = []

    def step(self, side, policy, value):
        leaf = self.root.select_leaf()
        leaf.expand(policy)
        leaf.backup(side, policy, value)

    def select_most(self):
        return self.root.select_most().action

    def place(self, policy, action):
        if action not in self.root.childs:
            self.root.expand(policy)

            if action not in self.root.childs:
                raise RuntimeError('given action is not legal')

        probs = torch.softmax(torch.Tensor([self.root.childs[action].visited /
                                            self.root.visited if action in self.root.childs else 0 for action in Game.all_actions()]))

        self.histories.append((self.root.state, probs))

        self.root = self.root.childs[action]
        self.root.parent = None
        self.root.action = None
