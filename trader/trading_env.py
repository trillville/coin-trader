from random import random
import pandas as pd
import numpy as np

from tensorforce import util, TensorForceError
from tensorforce.environments import Environment


class TradingEnv(Environment):

    def __init__(self, specification):
        """
        Skeleton of the GDAX trading env
        """
        self.actions_ = dict(
                action=dict(type='int', shape=(), num_actions=3), #buy BTC/sell BTC/hold (do nothing)
                amount=dict(type='float', shape=(), min_value=self.min_trade, max_value=max_trade)) #trade size

        self.states_ = dict(
            timeseries=dict(type='float', shape=self.train_data),  # time dependent features
            stationary=dict(type='float', shape=2)  # btc/eth holdings
        )

    def __str__(self):
        return 'TradingEnv'

    def close(self):
        pass

    def reset(self):
        self.state = {action_type: (1.0, 0.0) for action_type in self.specification}
        if self.single_state_action:
            return next(iter(self.state.values()))
        else:
            return dict(self.state)

    def execute(self, actions):
        if self.single_state_action:
            actions = {next(iter(self.specification)): actions}

        reward = 0.0
        for action_type, shape in self.specification.items():
            if action_type == 'bool' or action_type == 'int':
                correct = np.sum(actions[action_type])
                overall = util.prod(shape)
                self.state[action_type] = ((overall - correct) / overall, correct / overall)
            elif action_type == 'float' or action_type == 'bounded':
                step = np.sum(actions[action_type]) / util.prod(shape)
                self.state[action_type] = max(self.state[action_type][0] - step, 0.0), min(self.state[action_type][1] + step, 1.0)
            reward += max(min(self.state[action_type][1], 1.0), 0.0)

        terminal = random() < 0.25
        if self.single_state_action:
            return next(iter(self.state.values())), terminal, reward
        else:
            reward = reward / len(self.specification)
            return dict(self.state), terminal, reward

    @property
    def states(self):
        return self.states_

    @property
    def actions(self):
        return self.actions_