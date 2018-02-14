from random import random
import pandas as pd
import numpy as np
from collections import Counter
from math import floor, ceil
import time

from tensorforce.environments import Environment
from tensorforce.execution import Runner

class TradingEnv(Environment):
    """
    Skeleton of the GDAX trading env
    """
    def __init__(self, data, params):
        self.data = data # pandas dataframe (currently read from CSV)
        self.params = params

        # normalize TS data
        # self.data[:,6] = self.data[:,6] - self.data[:,6].min()
        # self.data[:,6] = self.data[:,6] / self.data[:,6].max()

        self.MIN_TRADE = 50.0
        self.MAX_TRADE = 10000.0

        self.START_USD = 10000.0 # number of starting usder
        self.START_BTC = 10000.0 # number of starting bitcoins
        self.FEE = 0.00 # exchange fee
        self.MAX_HOLD_LENGTH = 24 * 60 * 2 # punish if we don't do anything for a day

        self.TRAIN_TEST_SPLIT = 0.8 # 80% train, 20% test

        if params['single_action']:
            self._actions = dict(type='float', shape=(), min_value=-self.MAX_TRADE, max_value=self.MAX_TRADE)
        else:
            self._actions = dict(
                action=dict(type='int', shape=(), num_actions=3), # buy BTC/sell BTC/hold (do nothing)
                amount=dict(type='float', shape=(), min_value=self.MIN_TRADE, max_value=self.MAX_TRADE)
            )

        self._states = dict(
            env=dict(type='float', shape=self.data.shape[1]),  # environment states
            agent=dict(type='float', shape=4)  # agent states [usd, btc, repeats, last signal]
        )

        self.acc = dict(
            episode=dict(
                i=0,
                num_steps=0,
                advantages=[],
                uniques=[]
            ),
            step = dict(
                i=0,
                usd=self.START_USD,
                btc=self.START_BTC,
                val=self.START_USD + self.START_BTC,
                repeats=0,
                signals=[]
            )
        )

        self.hold = dict(usd=self.START_USD, btc=self.START_BTC) # this will be our baseline

    def __str__(self):
        return 'TradingEnv'

    def close(self):
        pass

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        self.hold['btc'] = self.START_USD + self.START_BTC

        ep_acc, step_acc = self.acc['episode'], self.acc['step']
        step_acc['usd'], step_acc['btc'] = self.START_USD, self.START_BTC
        step_acc['i'] = 0
        step_acc['signal'] = [0] * self.run_data.shape[1]
        step_acc['repeats'] = 0
        ep_acc['i'] += 1

        return self._get_next_state(self.START_USD, self.START_BTC, 0, 0)

    def _get_next_state(self, usd, btc, repeats, signal):
        env = self.data[self.acc['step']['i'],:].tolist()
        agent = [usd, btc, repeats, signal]

        return dict(env=env, agent=agent)

    def execute(self, actions):
        if self.params['single_action']:
            signal = 0 if -self.MIN_TRADE < actions < self.MIN_TRADE else actions
        else:
            signal = {
                0: -1,  # make amount negative
                1: 0,  # hold
                2: 1  # make amount positive
            }[actions['action']] * actions['amount']
            if not signal:
                signal = 0

        abs_sig = abs(signal)

        step_acc, ep_acc = self.acc['step'], self.acc['episode']

        # initialize last/curr usd/btc values and number of repeated actions
        last_usd, last_btc, last_signal = step_acc['usd'], step_acc['btc'], step_acc['signal']

        if signal > 0 and not abs_sig > last_usd:
            step_acc['btc'] += (1.0 - self.FEE) * abs_sig
            step_acc['usd'] -= abs_sig
        elif signal < 0 and not abs_sig > last_btc:
            step_acc['btc'] -= abs_sig
            step_acc['usd'] += (1.0 - self.FEE) * abs_sig

        step_acc['signal'] = signal
        step_acc['i'] += 1

        # Treating a buy and hold strategy as the baseline
        pct_change = self.price_pct_changes[step_acc['i']]
        step_acc['btc'] += pct_change * step_acc['btc']
        total = step_acc['usd'] + step_acc['btc']

        # see how the buy and hold strategy would have done
        hold = self.hold
        hold['btc'] += pct_change * hold['btc']
        reward = (total - hold['btc']) / total

        # Collect repeated same-action count (homogeneous actions punished below)
        recent_actions = np.array(step_acc['signals'][-step_acc['repeats']:])
        if np.any(recent_actions > 0) and np.any(recent_actions < 0) and np.any(recent_actions == 0):
            step_acc['repeats'] = 1  # reset repeat counter
        else:
            step_acc['repeats'] = 1

        #if step_acc['i'] % 1000 == 0:
        #    print("step: ", step_acc['i'], "usd: ", step_acc['usd'], "btc: ", step_acc['btc'], "tot val: ", total, "reward:", reward, "repeats: ", step_acc['repeats'], "signal: ", signal)

        # this step has to be after the time increment!
        next_state = self._get_next_state(step_acc['usd'], step_acc['btc'], step_acc['repeats'], signal)
        step_acc['signal'] = signal

        terminal = step_acc['i'] >= len(self.run_data) - 1
        if step_acc['repeats'] >= self.MAX_HOLD_LENGTH:
            reward -= -1.0  # start trading u cuck
            terminal = True

        if terminal is True:
            self.episode_finished()

        return next_state, terminal, reward

    def episode_finished(self):
        step_acc, ep_acc = self.acc['step'], self.acc['episode']
        signals = step_acc['signals']

        advantage = (step_acc['btc'] + step_acc['usd']) - (self.hold['btc'])

        ep_acc['advantages'].append(advantage)
        n_uniques = float(len(np.unique(signals)))
        ep_acc['uniques'].append(n_uniques)

        # Print (limit to note-worthy)
        common = dict((round(k,2), v) for k, v in Counter(signals).most_common(5))
        completion = f"|{int(ep_acc['num_steps'] / self.run_data.shape[0] * 100)}%"
        print("hold BTC: ", self.hold['btc'], " trader value: ", step_acc['btc'] + step_acc['usd'])

        print(f"{ep_acc['i']}|⌛:{step_acc['i']}{completion}\tA:{'%.3f'%advantage}\t{common}({n_uniques}uniq)")

        return True

    ## TODO: setup the run_sim and train_and_test functions. goal -> iterate through the dataset, making num_tests number of unique datasets
    ## for each dataset, do a train/test exercise and calculate/log the advantage
    def run_sim(self, runner, print_results=True):
        next_state, terminal = self.reset(), False
        while not terminal:
            next_state, terminal, reward = self.execute(runner.agent.act(next_state, deterministic=True))
        if print_results: self.episode_finished()

    def train_and_test(self, agent, num_runs):
        runner = Runner(agent=agent, environment=self)
        i = 0
        train, test = self.initialize_data()
        run_length = 24*60 # one week
        start_index = 0
        try:
            while i <= num_runs:
                print(test.shape[0])
                if start_index + run_length >= train.shape[0] - 1:
                    start_index = 0
                self.run_data = train[start_index:start_index + run_length, ]
                print("START INDEX: ", start_index)
                runner.run(timesteps=self.run_data.shape[0], max_episode_timesteps=self.run_data.shape[0])
                self.run_data = test
                self.run_sim(runner)
                #self.run_data = test
                #runner.run(timesteps=test.shape[0], max_episode_timesteps=test.shape[0])
                #self.run_sim(runner, print_results=True)
                i += 1
                start_index += run_length
                print(i)
                #self.run_sim(runner, print_results=True)
        except KeyboardInterrupt:
            self.run_data = test
            runner.run(max_episode_timesteps=test.shape[0])
            # Lets us kill training with Ctrl-C and skip straight to the final test. This is useful in case you're
            # keeping an eye on terminal and see "there! right there, stop you found it!" (where early_stop & n_steps
            # are the more musdodical approaches)
            pass

    def _get_diff(self, col, percent=False):
        series = pd.Series(col)
        diff = series.pct_change() if percent else series.diff()
        diff.iloc[0] = 0  # always NaN, nothing to compare to

        # Remove outliers (turn them to NaN)
        q = diff.quantile(0.99)
        diff = diff.mask(diff > q, np.nan)

        # then forward-fill the NaNs.
        return diff.replace([np.inf, -np.inf], np.nan).ffill().bfill().values

    def initialize_data(self):
        """
        Fetch and transform train/test data. Note - we are not using random sampling because it is timeseries data
        """
        self.price_pct_changes = self._get_diff(self.data[:,2], True)
        row_cnt = self.data.shape[0]
        n_train = ceil(row_cnt * self.TRAIN_TEST_SPLIT)
        train, test = self.data[0:n_train,], self.data[n_train:row_cnt,]
        return train, test

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions
