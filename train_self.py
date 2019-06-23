import os
import myenv
import gym
from myenv.RL_model import DQNmodel

TRAINING_STEP  = 100000
BEST_CHECK_NUM = 1000
# update best_model  if win_rate > 0.6
UPDATE_WIN_RATE = 0.53

BEST_WEIGHT_PATH = 'myenv/params/best_weights.h5f'
SAVE_WEIGHT_PATH = 'myenv/params/self{}_weights.h5f'

env = gym.make('othello_self-v0')
input_size = env.observation_space.shape
output_size = env.action_space.n
dqn_model = DQNmodel(input_size=input_size,output_size=output_size)

for i in range(1000):
    if not os.path.isfile(SAVE_WEIGHT_PATH.format(i)):
        update_count = i
        break

for i in range(1000):
    env.update_weights(BEST_WEIGHT_PATH)
    dqn_model.dqn.load_weights(BEST_WEIGHT_PATH)

    dqn_model.dqn.fit(env, nb_steps=TRAINING_STEP , visualize=False, verbose=1)

    win_count = 0
    for _ in range(BEST_CHECK_NUM):
        observation = env._reset()
        done = False
        while not done :
            action = dqn_model.dqn.forward(observation)
            observation,reward,done,info = env._step(action)
        if reward > 0:
            win_count += 1
    win_rate = win_count/(BEST_CHECK_NUM+1)

    update_flg = win_rate > UPDATE_WIN_RATE

    if update_flg:
        dqn_model.dqn.save_weights(SAVE_WEIGHT_PATH.format(update_count), overwrite=True)
        dqn_model.dqn.save_weights(BEST_WEIGHT_PATH, overwrite=True)
        update_count += 1

    print()
    print(str(i)+'finished \t best_change:'+str(update_flg)+'\t win_rate:'+str(win_rate))
    print()
