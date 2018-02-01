from random import random
from box import Box
import pandas as pd
import numpy as np

from tensorforce import util, TensorForceError
from tensorforce.environments import Environment


class TradingEnv(Environment):

    def __init__(self, data):
        """
        Skeleton of the GDAX trading env
        """
        self.data = data # pandas dataframe (currently read from CSV)
        self.start_timestep = data['open_time'].min()

        MIN_TRADE = 0.01
        MAX_TRADE = 0.1

        self.START_ETH = 10.0
        self.START_BTC = 1.0
        self.FEE = 0.003

        self.step = 0

        self._actions = dict(
                action=dict(type='int', shape=(), num_actions=3), # buy BTC/sell BTC/hold (do nothing)
                amount=dict(type='float', shape=(), min_value=0.01, max_value=0.1)) # trade size

        self._states = dict(
            env=dict(type='float', shape=self.data),  # environment states (independent of agent behavior)
            stationary=dict(type='float', shape=3)  # agent states (dependent on agent behavior) [eth, btc, repeats]
        )

    def __str__(self):
        return 'TradingEnv'

    def close(self):
        pass

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        self.step = 0
        return self._get_next_state(self.start_timestep, self.START_ETH, self.START_BTC)

    def _get_next_state(self, i, eth, btc):
        timeseries = self.data.iloc[i]
        stationary = [eth, btc]
        return dict(series=timeseries, stationary=stationary)

    def _calc_delta(self, column):
        diff = self.data[column].pct_change()
        diff.iloc[0] = 0  # get rid of nan
        return diff[self.step]

    def execute(self, actions):
        signal = {
            0: -1,  # make amount negative
            1: 0,  # hold
            2: 1  # make amount positive
        }[actions['action']] * actions['amount']
        if not signal: signal = 0
        abs_sig = abs(signal)

        reward = 0 # initialize reward
        last_eth, last_btc = self._states['stationary'] # initialize last/curr eth/btc values
        last_price = self.data['open'][self.step] # initialize price
        last_btc_value = last_btc + last_eth / last_price

        if signal > 0 and not (abs_sig > last_eth):
            curr_btc = last_btc + abs_sig - (abs_sig * fee)
            curr_eth = last_eth - abs_sig / last_price
        elif signal < 0 and not (abs_sig > last_btc):
            new_btc = last_btc - abs_sig
            new_eth = last_eth + (abs_sig - abs_sig * fee) / last_price

        # now increment time
        self.step += 1
        curr_price = self.data['open'][self.step]
        curr_btc_value = curr_btc + curr_eth / curr_price # value of action
        hold_btc_value = last_btc + last_eth / curr_price # value of doing nothing

        action_reward = curr_btc_value - last_btc_value # reward of action
        hold_reward = hold_btc_value - last_btc_value # reward of doing nothing

        return action_reward, hold_reward

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions