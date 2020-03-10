
import torch

from actor_critic import ActorCritic
from game import Game
from state import BLACK, WHITE, BOARD_SIZE

ai = ActorCritic()

ai.load('third')
print('ai loaded!')

state = Game.init()
ai.new_game()

turn = 0

while True:
    turn += 1

    print('turn #{}:'.format(turn))
    print(state)

    if state.turn == WHITE:  # User's turn!
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

        for _ in range(50):
            ai.step()

        action = ai.select_action()

    state = Game.next_state(state, action)
    ai.place(action)

    if state.is_terminated:
        if abs(state.reward) < .5:
            print('draw!')
        elif state.turn == WHITE:
            print('you lose!')
        else:
            print('you win!')

        print(state)

        break
