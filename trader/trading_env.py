from random import random
import pandas as pd
import numpy as np

from tensorforce.environments import Environment


class TradingEnv(Environment):
    """
    Skeleton of the GDAX trading env
    """
    def __init__(self, data):
        self.data = data # pandas dataframe (currently read from CSV)

        # normalize TS data
        self.data[:,0] = self.data[:,0] - self.data[:,0].min()
        self.data[:,0] = self.data[:,0] / self.data[:,0].max()

        self.data[:,6] = self.data[:,6] - self.data[:,6].min()
        self.data[:,6] = self.data[:,6] / self.data[:,6].max()

        MIN_TRADE = 0.01
        MAX_TRADE = 0.1

        self.START_ETH = 10.0 # number of starting ether
        self.START_BTC = 1.0 # number of starting bitcoins
        self.FEE = 0.003 # exchange fee
        self.MAX_HOLD_LENGTH = 500 # punish if we don't do anything for 500 consecutive timesteps

        self._actions = dict(
            action=dict(type='int', shape=(), num_actions=3), # buy BTC/sell BTC/hold (do nothing)
            amount=dict(type='float', shape=(), min_value=MIN_TRADE, max_value=MAX_TRADE)
        )

        self._states = dict(
            env=dict(type='float', shape=self.data.shape[1]),  # environment states
            agent=dict(type='float', shape=4)  # agent states [eth, btc, repeats, last signal]
        )

        self.acc = dict(
            step=0,
            eth=self.START_ETH,
            btc=self.START_BTC,
            repeats=0,
            signal=0
        )

    def __str__(self):
        return 'TradingEnv'

    def close(self):
        pass

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        self.acc = dict(
            step=0,
            eth=self.START_ETH,
            btc=self.START_BTC,
            repeats=0,
            signal=0
        )

        return self._get_next_state(self.START_ETH, self.START_BTC, 0, 0)

    def _get_next_state(self, eth, btc, repeats, signal):
        env = self.data[self.acc['step'],:].tolist()
        agent = [eth, btc, repeats, signal]

        return dict(env=env, agent=agent)

    def execute(self, actions):
        signal = {
            0: -1,  # make amount negative
            1: 0,  # hold
            2: 1  # make amount positive
        }[actions['action']] * actions['amount']
        if not signal:
            signal = 0
        abs_sig = abs(signal)

        reward = 0 # initialize reward

        # initialize last/curr eth/btc values and number of repeated actions
        last_eth, last_btc, last_signal = self.acc['eth'], self.acc['btc'], self.acc['signal']
        last_price = self.data[self.acc['step'],1] # initialize price
        last_btc_value = last_btc + last_eth / last_price

        if signal > 0 and not abs_sig > last_eth:
            self.acc['btc'] = last_btc + abs_sig - (abs_sig * self.FEE)
            self.acc['eth'] = last_eth - abs_sig / last_price
        elif signal < 0 and not abs_sig > last_btc:
            self.acc['btc'] = last_btc - abs_sig
            self.acc['eth'] = last_eth + (abs_sig - abs_sig * self.FEE) / last_price

        # now increment time (probably a better way to do this without using a global var...)
        self.acc['step'] += 1
        self.acc['signal'] = signal

        curr_price = self.data[self.acc['step'],1]

        # not sure if this is the best comparison... could also compare to reward of holding
        curr_btc_value = self.acc['btc'] + self.acc['eth'] / curr_price
        reward = curr_btc_value - last_btc_value

        # Collect repeated same-action count (homogeneous actions punished below)
        if signal == last_signal: # no change in signal
            self.acc['repeats'] += 1
        else:
            self.acc['repeats'] = 1

        if self.acc['step'] % 100 == 0:
            print("step: ", self.acc['step'], "eth: ", self.acc['eth'], "btc: ",self.acc['btc'], "tot val: ", curr_btc_value, "reward:", reward, "repeats: ", self.acc['repeats'], "signal: ", signal)

        # this step has to be after the time increment!
        next_state = self._get_next_state(self.acc['eth'], self.acc['btc'], self.acc['repeats'], signal)
        self.acc['signal'] = signal

        terminal = self.acc['step'] >= len(self.data) - 1
        if self.acc['repeats'] >= self.MAX_HOLD_LENGTH:
            reward -= -1.0  # start trading u cuck
            terminal = True

        return next_state, terminal, reward

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions
