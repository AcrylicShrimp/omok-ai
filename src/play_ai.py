
import random

import torch

from actor_critic import ActorCritic
from mcts_pure import MCTS
from game import Game
from state import BLACK, WHITE, BOARD_SIZE

ai = ActorCritic()
ai.load('third')

while True:
    mcts = MCTS()
    ai.new_game()

    state = Game.init()

    turn = 0

    ai_side = random.choice([BLACK, WHITE])

    while True:
        turn += 1

        if state.turn == ai_side:
            for _ in range(50):
                ai.step()

            action = ai.select_action()
        else:
            for _ in range(50):
                mcts.step()

            action = mcts.select_best()

        state = Game.next_state(state, action)
        ai.place(action)
        mcts.place(action)

        if state.is_terminated:
            if abs(state.reward) < .5:
                print('draw!')
            elif state.turn == ai_side:
                print('MCTS win!')
            else:
                print('ai win!')

            print(state)
            print()

            break
