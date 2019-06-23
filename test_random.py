import myenv

import numpy as np
import gym

from myenv.RL_model import DQNmodel

env = gym.make('othello_random-v0')

input_size = env.observation_space.shape
output_size = env.action_space.n

dqn_model = DQNmodel(input_size=input_size,output_size=output_size)

dqn_model.dqn.load_weights('myenv/params/random_weights.h5f')

win_count = 0
for i in range(1000):

    observation = env._reset()
    done = False
    while not done :
        action = dqn_model.dqn.forward(observation)
        observation,reward,done,info = env._step(action)

    if reward > 0:
        result = 'win '
        win_count += 1
    elif reward < 0:
        result = 'lose'
    else:
        result = 'draw'
    print(str(i)+'episode '+result+'\t   win_rate:'+str(win_count/(i+1))+'%')
