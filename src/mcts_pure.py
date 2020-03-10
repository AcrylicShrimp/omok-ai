

import math
import random

import torch

from game import Game


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state      # board
        self.n = 0              # num. of visits
        self.w = 0              # total reward(accumulative)
        self.q = 0              # average reward(w / n)
        self.childs = {}        # action -> child node
        self.parent = parent    # parent node
        self.action = action    # action (integer)

    @property
    def is_leaf(self):
        """
        Tests whether self is leaf or not.
        """
        return len(self.childs) == 0

    def calc_ucb(self, c_puct):
        """
        Calculates the UCB value of self.
        NOTE: We're reversing q value here,
        because the highest q value child leads thier parent node to take bad action.
        UCB = -q + U
        U = c_puct * sqrt(N / (1 + n))
        """
        return -self.q + c_puct * math.sqrt((self.parent.n if self.parent is not None else 0) / (1 + self.n))

    def select_best(self):
        """
        Returns a child node that has lowest q value.
        NOTE: We're searching for the lowest q value here,
        because the highest q value child leads thier parent node to take bad action.
        """
        if self.is_leaf:
            return None

        # Gets childs.
        childs = [(child.q, child)
                  for child in self.childs.values()]

        # Shuffle them - to randomly select a child when there're many childs that have same visit count.
        random.shuffle(childs)

        # Sort them and returns first child.
        return sorted(childs, key=lambda x: x[0])[0][1]

    def select_leaf(self):
        """
        Returns a child node that is leaf node.
        """
        if self.is_leaf:
            return self

        # Gets childs.
        childs = [(child.calc_ucb(1.), child)
                  for child in self.childs.values()]

        # Shuffle them - to randomly select a child when there're many childs that have same UCB value.
        random.shuffle(childs)

        # Sort them and returns "recursive" result.
        # IMPORTANT: Recursive call here!!!
        return sorted(childs, key=lambda x: x[0], reverse=True)[0][1].select_leaf()

    def expand(self):
        """
        Generates all possible childs if self is leaf node.
        """
        if not self.is_leaf:
            return

        if self.state.is_terminated:
            return

        actions = Game.possible_actions(self.state)

        # Gets all possible actions and creates child nodes for each action.
        for action in actions:
            self.childs[action] = \
                MCTSNode(
                    Game.next_state(self.state, action),    # next state
                    self,                                   # parent node
                    action)                                 # action

    def backup(self):
        """
        Simulates game once and propagate back the result.
        """
        state = self.state

        # Random rollout.
        while not state.is_terminated:
            state = Game.next_state(
                state,
                random.choice(Game.possible_actions(state)))

        node = self

        while node is not None:
            reward = state.reward * \
                (1 if state.turn == node.state.turn else -1)

            node.n += 1
            node.w += reward
            node.q = node.w / node.n
            node = node.parent


class MCTS:
    def __init__(self):
        self.root = MCTSNode(Game.init())

    def step(self):
        """
        Makes MCTS one step more.
        1. Selects a leaf.
        2. Expands it(if needed).
        3. Selects a leaf once more(because step #2 may expanded that's children).
        4. Simulates and back-ups the results.
        """
        leaf = self.root.select_leaf()
        leaf.expand()
        leaf = leaf.select_leaf()
        leaf.backup()

    def select_best(self):
        """
        Selects an available action that has the lowest Q value.
        """
        return self.root.select_best().action

    def place(self, action):
        """
        Moves root node and detaches it.
        """
        if action not in self.root.childs:
            self.root.expand()

            if action not in self.root.childs:
                raise RuntimeError('given action is not legal')

        self.root = self.root.childs[action]
        self.root.parent = None
        self.root.action = None
