import matplotlib
matplotlib.use('Agg')

import os

from functools import partial
from datetime import datetime

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

from multiprocessing import Pool
import itertools

import tensorflow as tf

from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy, LnCnnPolicy, LnMlpPolicy
from stable_baselines import DQN
from stable_baselines.common.atari_wrappers import make_atari




def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


n_steps = 0

def callback(_locals, _globals):
    global n_steps
    print(n_steps)
    n_steps += 1
    return False

def run_experiment(params):
    os.makedirs('/home/ubuntu/gym/', exist_ok=True)
    with open("/home/ubuntu/gym/ta_experiments.log", "a") as myfile:
        t = datetime.now()
        myfile.write('start: at {} \t with {}\n'.format(t, params))   

    experiment_name = params['experiment_name']
    env_name = params['env_name']
    noise_method = params['noise_method']
    seed = params['seed']


    policy = params['policy']
    total_timesteps = params['total_timesteps']

    # Create log dir
    log_dir = f"/home/ubuntu/gym/{experiment_name}/{env_name}/{noise_method}/{str(seed)}/"
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('CartPole-v1')
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    if noise_method == 'asn':
        param_noise = False
        policy = partial(policy, dueling=False, buffer_size=1)

    if noise_method == 'psn':
        param_noise = True
        policy = partial(policy, dueling=False, buffer_size=1)

    model = DQN(
        policy,
        env,
        param_noise=param_noise,
        prioritized_replay=False,
        # verbose=0
    )
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        # callback=callback,
        seed=seed
    )

    return params

if __name__ == "__main__":

    p = Pool(1)

    d = {
        'experiment_name': ['ta_e04'],
        'env_name': [ 'Pendulum-v0'],
        'noise_method': ['asn', 'psn'],
        'seed': list(range(1,21)),
        'total_timesteps': [25_000],
        'policy': [MlpPolicy],

    }

    iterator = p.imap(run_experiment, dict_product(d))

    for res in iterator:
        print('done with:')
        print(res)
        with open("/home/ubuntu/gym/ta_experiments.log", "a") as myfile:
            t = datetime.now()
            myfile.write('done: at {} \t with {}\n'.format(t, res))
