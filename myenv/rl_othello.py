from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy,GreedyQPolicy

from myenv.OthelloFunction import onehot_to_board,get_possible_actions

import numpy as np

class OthelloDQNAgent(DQNAgent):
    def __init__(self, model, policy=None, test_policy=None, enable_double_dqn=False, enable_dueling_network=False,
                 dueling_type='avg', *args, **kwargs):

        super(OthelloDQNAgent, self).__init__(model,*args, **kwargs)

        self.policy = my_EpsGreedyQPolicy()
        self.test_policy = my_GreedyQPolicy()


    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        q_values = self.compute_q_values(state)
        if self.training:
            action = self.policy.select_action(q_values=q_values,obs=observation)
        else:
            action = self.test_policy.select_action(q_values=q_values,obs=observation)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

class my_EpsGreedyQPolicy(EpsGreedyQPolicy):
    def select_action(self, q_values,obs):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        board = onehot_to_board(obs)
        action_list = get_possible_actions(board,0)

        if action_list ==  []:
            action = 64 # pass

        elif np.random.uniform() < self.eps:
            action = np.random.choice(action_list)

        else:
            desce_q = sorted(enumerate(q_values), key=lambda x: -x[1])
            for i in range(66):
                if desce_q[i][0] in action_list:
                    action = desce_q[i][0]
                    break
        return action

class my_GreedyQPolicy(GreedyQPolicy):
    """Implement the greedy policy
    Greedy policy returns the current best action according to q_values
    """
    def select_action(self, q_values,obs):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        board = onehot_to_board(obs)
        action_list = get_possible_actions(board,0)

        if action_list ==  []:
            return 64 # pass

        desce_q = sorted(enumerate(q_values), key=lambda x: -x[1])
        for i in range(65):
            if desce_q[i][0] in action_list:
                action = desce_q[i][0]
                break
        return action
