import myenv
import gym

from myenv.RL_model import DQNmodel

env = gym.make('othello_random-v0')

input_size = env.observation_space.shape
output_size = env.action_space.n

dqn_model = DQNmodel(input_size=input_size,output_size=output_size)

dqn_model.dqn.fit(env, nb_steps=100000, visualize=False, verbose=1)

dqn_model.dqn.save_weights('myenv/params/random_weights.h5f', overwrite=True)

dqn_model.dqn.test(env, nb_episodes=1, visualize=True)
