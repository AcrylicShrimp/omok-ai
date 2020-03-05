
BLACK = -1
WHITE = 1


class State:
    def __init__(self, turn, state):
        self.turn = turn
        self.state = state
        self.reward = None

    def __str__(self):
        results = []

        for y in range(9):
            for x in range(9):
                me = self.state[self.get_turn_index(self.turn)][x + y * 9]
                opposite = self.state[self.get_turn_index(
                    -self.turn)][x + y * 9]

                if me > .5:
                    results.append('<>')
                elif opposite > .5:
                    results.append('[]')
                else:
                    results.append('--')

            results.append('\n')

        results.pop()

        return ''.join(results)

    @property
    def is_terminated(self):
        return self.reward is not None

    def get_turn_index(self, turn):
        return 0 if self.turn == turn else 1

    def is_legal(self, action):
        return self.state[2][action] > .5

    def place_dot(self, turn, action):
        if self.is_legal(action):
            self.state[self.get_turn_index(turn)][action] = 1.
            self.state[2][action] = .0

    def trace_dot(self, turn, begin_action, cb):
        count = 0
        index = begin_action

        while True:
            dot = self.state[self.get_turn_index(turn)][index]

            if dot < .5:
                break

            count += 1
            x, y = cb(index % 9, index // 9)

            if x is None:
                break

            if y is None:
                break

            if x < 0 or x >= 9:
                break

            if y < 0 or y >= 9:
                break

            index = x + y * 9

        return count
