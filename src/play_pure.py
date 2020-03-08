
import torch

from mcts_pure import MCTS
from game import Game
from state import BLACK, WHITE, BOARD_SIZE

mcts = MCTS()
print('tree ready!')

state = Game.init()

turn = 0

while True:
    turn += 1

    print('turn #{}:'.format(turn))
    print(state)

    if state.turn == BLACK:  # User's turn!
        while True:
            action = input('where you want to place your dot?')

            try:
                action = int(action)
            except:
                continue

            if action < 0:
                continue

            if action >= BOARD_SIZE ** 2:
                continue

            if not state.is_legal(action):
                continue

            break

    else:
        print('ai is thinking...')

        for _ in range(100):
            mcts.step()

        action = mcts.select_most()

    state = Game.next_state(state, action)
    mcts.place(action)

    if state.is_terminated:
        if state.turn == BLACK:
            print('you lose!')
        else:
            print('you win!')

        print(state)

        break
