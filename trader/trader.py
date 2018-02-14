from trader.trading_env import TradingEnv
import pandas as pd
import numpy as np
import json

from argparse import ArgumentParser
from tensorforce.agents import PPOAgent, RandomAgent
from tensorforce.core.networks import LayeredNetwork
from tensorforce.execution import Runner

import tensorflow as tf
import logging

def main(args):
    def build_net_spec():
        """Builds an array of dicts that conform to TForce's network specification (see their docs) by mix-and-matching
        different network hypers
        """

        dense = {
            'type': 'dense',
            'activation': 'relu',
            'l2_regularization': 0.01,
            'l1_regularization': 0.01
        }

        arr = []

        for i in range(1,2):
            if i == 1:
                size = 8
            elif i == 2:
                size = 16
            elif i == 3:
                size = 8
            arr.append({'size': size, **dense})

        print(arr)

        return arr

    def custom_net():
        layers_spec = build_net_spec()

        class CustomNet(LayeredNetwork):
            """
            https://github.com/reinforceio/tensorforce/blob/f0876f55e2cfb0af789479237bed7d8fe66df515/tensorforce/tests/test_tutorial_code.py
            """
            def __init__(self, **kwargs):
                super(CustomNet, self).__init__(layers_spec, **kwargs)

            def tf_apply(self, x, internals, update, return_internals=False):
                series = x['env']
                stationary = x['agent']
                x = series
                x = tf.concat([x, stationary], axis=1)

                # apply_stationary_here = 10
                internal_outputs = list()
                index = 0

                for i, layer in enumerate(self.layers):
                    layer_internals = [internals[index + n] for n in range(layer.num_internals)]
                    index += layer.num_internals
                    # if i == apply_stationary_here:
                    #     x = tf.concat([x, stationary], axis=1)
                    x = layer.apply(x, update, *layer_internals)

                    if not isinstance(x, tf.Tensor):
                        internal_outputs.extend(x[1])
                        x = x[0]

                if return_internals:
                    return x, list()
                else:
                    return x
        return CustomNet

    sample_data = pd.read_csv(args.data_path).values

    env = TradingEnv(data=sample_data, params=dict(single_action=True))
    network = custom_net()

    test_random = False

    if test_random == True:
        agent = RandomAgent(
            states_spec = env.states,
            actions_spec = env.actions)
    else:
        agent = PPOAgent(
            states_spec=env.states,
            actions_spec=env.actions,
            network_spec=network,
            batch_size=32,
            # BatchAgent
            keep_last_timestep=True,
            # PPOAgent
            step_optimizer=dict(
                type='adam',
                learning_rate=3e-4
            ),
            optimization_steps=10,
            # Model
            scope='ppo',
            discount=0.95,
            # DistributionModel
            distributions_spec=None,
            entropy_regularization=0.01,
            # PGModel
            baseline_mode=None,
            baseline=None,
            baseline_optimizer=None,
            gae_lambda=None,
            # PGLRModel
            likelihood_ratio_clipping=0.2,
            summary_spec=None,
            distributed_spec=None
        )


    runner = Runner(agent, env)

    report_episodes = 1

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode, ts=r.timestep))
            print("Episode reward: {}".format(r.episode_rewards[-1]))
            print("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
            print("Current state information: ", r.environment.acc['step']['val'])
        return True

    print("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))

    max_episodes = 10000
    max_timesteps = 1000

    runner.run(max_episodes, max_timesteps, episode_finished=episode_finished)

    print("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data_path',
                        required=False, default="/Users/tillmanelser/crypto/trading/coin-bot/sample_data/BTC_Med.csv")
    parser.add_argument('-n', '--network_spec', dest='network_spec',
                        required=False, default="/Users/tillmanelser/crypto/trading/coin-bot/trader/network_spec.json")
    return parser

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    main(args)
