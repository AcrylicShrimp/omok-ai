
import torch

from actor_critic import ActorCritic
from mcts_pure import MCTS
from game import Game
from state import BLACK, WHITE, BOARD_SIZE

ai = ActorCritic()
mcts = MCTS()

ai.load('second')
print('ai loaded!')

state = Game.init()
ai.new_game()

turn = 0

while True:
    turn += 1

    print('turn #{}:'.format(turn))
    print(state)

    if state.turn == WHITE:  # MCTS's turn!
        for _ in range(100):
            mcts.step()

        action = mcts.select_most()

    else:
        for _ in range(100):
            ai.step()

        action = ai.select_action()

    state = Game.next_state(state, action)
    mcts.place(action)
    ai.place(action)

    if state.is_terminated:
        if abs(state.reward) < .5:
            print('draw!')
        elif state.turn == WHITE:
            print('MCTS win!')
        else:
            print('AC win!')

        print(state)

        break
