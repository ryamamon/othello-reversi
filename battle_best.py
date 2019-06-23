import myenv

import numpy as np
import gym

from myenv.RL_model import DQNmodel
from myenv.OthelloFunction import action_to_point,_to_action, get_possible_actions, onehot_to_board

ENV_WEIGHT_PATH = 'myenv/params/best_weights.h5f'

env = gym.make('othello_self-v0')

env.update_weights(ENV_WEIGHT_PATH)

observation = env._reset()
done = False
while not done :
    env._render()

    board = onehot_to_board(observation)
    action_list = get_possible_actions(board,0)

    if action_list == []:
        action_list.append(64)

    put_point = [0,0]
    while True:
        point = input('please input putting point x,y >>')
        point = point.split(',')
        put_point[0] = int(point[1]) - 1
        put_point[1] = int(point[0]) - 1
        action = point_to_action(put_point)
        if action in action_list:
            break

    observation,reward,done,info = env._step(action)

if reward > 0:
    result = 'win '
elif reward < 0:
    result = 'lose'
else:
    result = 'draw'

print('your '+result+'!!!')
