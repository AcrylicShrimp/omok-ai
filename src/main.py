
import torch

from actor_critic import ActorCritic
from game import Game
from state import BLACK


ai = ActorCritic()


def step():
    state = Game.init()
    ai.new_game()

    while True:
        for _ in range(10):
            ai.step(state.turn)

        action = ai.select_action()
        state = Game.next_state(state, action)
        ai.place(action)

        print(state)
        print()

        if state.is_terminated:
            print('reward:', state.reward)
            ai.train(state)
            break


count = 0

while True:
    step()
    count += 1

    if count == 50:
        count = 0
        ai.save('first')
