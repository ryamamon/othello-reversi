import gym
import sys
import random
import copy
import numpy as np
from myenv.OthelloFunction import check_finished, do_action,get_onehot,get_reward

from myenv.RL_model import DQNmodel

NO_STONE = -1
BLACK = 0
WHITE = 1
BOARD_SIZE = 8

WEIGHT_PATH = 'myenv/params/best_weights.h5f'

class OthelloSelfEnv(gym.Env):

    def __init__(self):
        super().__init__()
        # 0~63:put stone 64:pass
        self.action_space = gym.spaces.Discrete(65)
        self.observation_space = gym.spaces.Box(
            low = -1,
            high = 2,
            shape = (8,8,2)
        )
        self.reward_range = [-100,100]

        self.vsModel = DQNmodel()
        self.vsModel.dqn.load_weights(WEIGHT_PATH)

    def _reset(self):
        self.board = np.ones((BOARD_SIZE,BOARD_SIZE)) * -1
        self.board[3,3] = BLACK
        self.board[3,4] = WHITE
        self.board[4,3] = WHITE
        self.board[4,4] = BLACK

        # setting hand

        # env is first hand
        if random.random() >= 0.5:
            self.env_hand = BLACK
            self.agt_hand = WHITE

            self.env_board_onehot = get_onehot(self.board,self.env_hand)
            action = self.vsModel.dqn.forward(self.env_board_onehot)
            self.board = do_action(self.board,self.env_hand,action)

        # env is late hand
        else:
            self.env_hand = WHITE
            self.agt_hand = BLACK

        self.env_board_onehot = get_onehot(self.board,self.env_hand)
        self.agt_board_onehot = get_onehot(self.board,self.agt_hand)
        self.done = False

        return self.agt_board_onehot

    def _step(self, action):

        """ agent put stone """
        self.board = do_action(self.board,self.agt_hand,action)
        self.env_board_onehot = get_onehot(self.board,self.env_hand)
        self.agt_board_onehot = get_onehot(self.board,self.agt_hand)

        if check_finished(self.board):
            reward = get_reward(self.board,self.agt_hand)
            return self.agt_board_onehot, reward, True, {}

        """ enviroment put stone """
        action = self.vsModel.dqn.forward(self.env_board_onehot)
        self.board = do_action(self.board,self.env_hand,action)
        self.env_board_onehot = get_onehot(self.board,self.env_hand)
        self.agt_board_onehot = get_onehot(self.board,self.agt_hand)

        if check_finished(self.board):
            reward = get_reward(self.board,self.agt_hand)
            return self.agt_board_onehot, reward, True, {}

        return self.agt_board_onehot, 0, False, {}

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        outfile.write(' ' * 6)
        for j in range(self.board.shape[1]):
            outfile.write(' ' +  str(j + 1) + '   | ')
        outfile.write('\n')
        outfile.write(' ' * 5)
        outfile.write('-' * (self.board.shape[1] * 6 - 1))
        outfile.write('\n')
        for i in range(self.board.shape[1]):
            outfile.write(' ' +  str(i + 1) + '  |')
            for j in range(self.board.shape[1]):
                if self.board[i, j] == NO_STONE:
                    outfile.write('  　  ')
                elif self.board[i, j] == BLACK:
                    outfile.write('  ●   ')
                else:
                    outfile.write('  ◯ 　')
                outfile.write('|')
            outfile.write('\n')
            outfile.write(' ' )
            outfile.write('-' * (self.board.shape[1] * 7 - 1))
            outfile.write('\n')

        if mode != 'human':
            return outfile

    def _close(self):
        pass

    def _seed(self, seed=None):
        pass

    def update_weights(self,path=WEIGHT_PATH):
        self.vsModel.dqn.load_weights(path)
