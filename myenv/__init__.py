from gym.envs.registration import register

register(
    id='othello_random-v0',
    entry_point='myenv.OthelloRandom:OthelloRandomEnv'
)

register(
    id='othello_self-v0',
    entry_point='myenv.OthelloSelf:OthelloSelfEnv'
)
