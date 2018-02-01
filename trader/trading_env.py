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

        self.START_ETH = 10.0 # number of starting ether
        self.START_BTC = 1.0 # number of starting bitcoins
        self.FEE = 0.003 # exchange fee
        self.MAX_HOLD_LENGTH = 500 # punish if we don't do anything for 500 consecutive timesteps

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
        return self._get_next_state(self.START_ETH, self.START_BTC)

    def _get_next_state(self, eth, btc):
        timeseries = self.data.iloc[self.step]
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
        last_eth, last_btc, repeats = self._states['stationary'] # initialize last/curr eth/btc values and number of repeated actions
        last_price = self.data['open'][self.step] # initialize price
        last_btc_value = last_btc + last_eth / last_price

        if signal > 0 and not (abs_sig > last_eth):
            curr_btc = last_btc + abs_sig - (abs_sig * fee)
            curr_eth = last_eth - abs_sig / last_price
        elif signal < 0 and not (abs_sig > last_btc):
            curr_btc = last_btc - abs_sig
            curr_eth = last_eth + (abs_sig - abs_sig * fee) / last_price

        self.step += 1 # now increment time (probably a better way to do this without using a global var...)
        curr_price = self.data['open'][self.step]

        reward = (curr_btc + curr_eth / curr_price) - last_btc_value # not sure if this is the best comparison... could also compare to reward of holding

        # Collect repeated same-action count (homogeneous actions punished below)
        if signal == self.last_signal:
            repeats += 1
        else:
            repeats = 1

        self.last_signal = signal

        next_state = self._get_next_state(curr_eth, curr_btc) # this step has to be after the time increment!

        terminal = self.step >= len(self.data)
        if repeats >= self.MAX_HOLD_LENGTH:
            reward -= -1.0  # start trading u cuck
            terminal = True
        if terminal:
            step_acc.signals.append(0)  # Add one last signal (to match length)

        # if step_acc.value <= 0 or step_acc.cash <= 0: terminal = 1
        return next_state, terminal, reward

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions