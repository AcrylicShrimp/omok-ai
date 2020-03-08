
import random

import torch

from actor_critic import ActorCritic
from mcts_pure import MCTS
from game import Game
from state import BLACK, WHITE


ai = ActorCritic()
# ai.load('second')


def step():
    mcts = MCTS()

    histories = []
    state = Game.init()
    ai.new_game()

    ai_side = random.choice([BLACK, WHITE])

    while True:
        if state.turn == ai_side:
            for _ in range(100):
                ai.step()
            for _ in range(100):
                mcts.step()

            action = ai.select_action()
        else:
            for _ in range(100):
                ai.step()
            for _ in range(100):
                mcts.step()

            action = mcts.select_most()

        state = Game.next_state(state, action)
        histories.append(ai.place(action))
        mcts.place(action)

        if state.is_terminated:
            print()
            print(state)
            print('reward:', state.reward
                  if state.turn == ai_side else -state.reward)

            ai.train(state, histories)

            break


count = 0

while True:
    # cProfile.run('step()', sort='tottime')
    step()

    count += 1

    if count == 10:
        count = 0
        ai.save('second')
        print('saved!')
