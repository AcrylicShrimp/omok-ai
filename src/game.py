
import torch

from state import BLACK, WHITE, State


class Game:
    @staticmethod
    def init():
        return State(BLACK, torch.cat([torch.zeros(2, 81), torch.ones(1, 81)]))

    @staticmethod
    def possible_actions(state):
        return [action for action in range(81) if state.state[2][action] > .5]

    @staticmethod
    def next_state(state, action):
        new_state = State(-state.turn,
                          torch.cat([state.state[1].unsqueeze(0), state.state[0].unsqueeze(0), state.state[2].unsqueeze(0)]))
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

        if left + right == 6:
            new_state.reward = -1
            return new_state

        if top + bottom == 6:
            new_state.reward = -1
            return new_state

        if topleft + bottomright == 6:
            new_state.reward = -1
            return new_state

        if topright + bottomleft == 6:
            new_state.reward = -1
            return new_state

        for action in range(81):
            if new_state.state[2][action] > .5:
                return new_state

        new_state.reward = 0  # Draw
        return new_state
