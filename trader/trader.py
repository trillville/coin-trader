from trader.trading_env import TradingEnv
import pandas as pd
import numpy as np
import json

from argparse import ArgumentParser
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.core.networks import LayeredNetwork
import tensorflow as tf

def main(args):
    def build_net_spec():
        """Builds an array of dicts that conform to TForce's network specification (see their docs) by mix-and-matching
        different network hypers
        """

        dense = {
            'type': 'dense',
            'activation': 'tanh',
            'l2_regularization': 0.0,
            'l1_regularization': 0.0
        }

        arr = []

        for i in range(1,3):
            size = 32
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

                apply_stationary_here = 1
                internal_outputs = list()
                index = 0

                for i, layer in enumerate(self.layers):
                    layer_internals = [internals[index + n] for n in range(layer.num_internals)]
                    index += layer.num_internals
                    if i == apply_stationary_here:
                        x = tf.concat([x, stationary], axis=1)
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

    env = TradingEnv(data=sample_data)
    network = custom_net()

    agent = PPOAgent(
        states_spec=env.states,
        actions_spec=env.actions,
        network_spec=network,
        batch_size=500,
        # BatchAgent
        keep_last_timestep=True,
        # PPOAgent
        step_optimizer=dict(
            type='adam',
            learning_rate=1e-3
        )
    )

    runner = Runner(agent=agent, environment=env)

    def episode_finished(runner):
        if runner.episode % 100 == 0:
            print(sum(runner.episode_rewards[-100:]) / 100)
        return runner.episode < 100 \
            or not all(reward > 0.0 for reward in runner.episode_rewards[-100:])

    runner.run(episodes=1000, episode_finished=episode_finished)

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data_path',
                        required=False, default="/Users/tillmanelser/crypto/trading/coin-bot/sample_data/sample_data.csv")
    parser.add_argument('-n', '--network_spec', dest='network_spec',
                        required=False, default="/Users/tillmanelser/crypto/trading/coin-bot/trader/network_spec.json")
    return parser

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    main(args)