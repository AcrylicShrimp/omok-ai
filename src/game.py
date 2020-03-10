
import torch

from state import BLACK, WHITE, BOARD_SIZE, State

END_SIZE = 3


class Game:
    @staticmethod
    def init():
        return State(BLACK, torch.cat([torch.zeros(2, BOARD_SIZE ** 2), torch.ones(1, BOARD_SIZE ** 2)]))

    @staticmethod
    def all_actions():
        return list(range(BOARD_SIZE ** 2))

    @staticmethod
    def possible_actions(state):
        return [action for action in range(BOARD_SIZE ** 2) if state.is_legal(action)]

    @staticmethod
    def next_state(state, action):
        new_state = State(-state.turn,
                          torch.cat([state.state[1].unsqueeze(0), state.state[0].unsqueeze(0), state.state[2].unsqueeze(0)]))

        assert state.turn != new_state.turn

        # IMPORTANT!
        # We're using state.turn instead of new_state.turn.
        new_state.place_dot(state.turn, action)

        def trace_dot(cb):
            return new_state.trace_dot(state.turn, action, cb)

        left = trace_dot(lambda x, y: (x - 1, y))
        right = trace_dot(lambda x, y: (x + 1, y))
        top = trace_dot(lambda x, y: (x, y - 1))
        bottom = trace_dot(lambda x, y: (x, y + 1))
        topleft = trace_dot(lambda x, y: (x - 1, y - 1))
        topright = trace_dot(lambda x, y: (x + 1, y - 1))
        bottomleft = trace_dot(lambda x, y: (x - 1, y + 1))
        bottomright = trace_dot(lambda x, y: (x + 1, y + 1))

        if left + right == END_SIZE + 1:
            new_state.reward = -1
            return new_state

        if top + bottom == END_SIZE + 1:
            new_state.reward = -1
            return new_state

        if topleft + bottomright == END_SIZE + 1:
            new_state.reward = -1
            return new_state

        if topright + bottomleft == END_SIZE + 1:
            new_state.reward = -1
            return new_state

        for action in range(BOARD_SIZE ** 2):
            if new_state.state[2][action] > .5:
                return new_state

        new_state.reward = 0  # Draw
        return new_state
