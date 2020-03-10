
import random

import torch

from actor_critic import ActorCritic
from game import Game


ai = ActorCritic()
# ai.load('second')


def step():
    histories = []
    state = Game.init()
    ai.new_game()

    while True:
        for _ in range(100):
            ai.step()

        action = ai.select_action()

        state = Game.next_state(state, action)
        histories.append(ai.place(action, True))

        if state.is_terminated:
            print()
            print(state)

            if abs(state.reward) < .5:
                print('draw!')
            else:
                print('not draw!')

            ai.train(state, histories)

            break


count = 0

while True:
    step()

    count += 1

    if count == 10:
        count = 0
        ai.save('third')
        print('========== saved! ==========')
