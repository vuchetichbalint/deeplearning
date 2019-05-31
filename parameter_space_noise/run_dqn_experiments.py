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
    log_dir = '/home/ubuntu/gym/'
    os.makedirs(log_dir, exist_ok=True)
    with open(log_dir + "dqn_experiments.log", "a") as myfile:
        t = datetime.now()
        myfile.write('start: at {} \t with {}\n'.format(t, params))   

    experiment_name = params['experiment_name']
    env_name = params['env_name'] + 'NoFrameskip-v4'
    noise_method = params['noise_method']
    seed = params['seed']


    policy = params['policy']
    total_timesteps = params['total_timesteps']

    # Create log dir
    # log_dir = "/root/code/stable-baselines/gym/{}/{}/{}/{}/".format(experiment_name, env_name, noise_method, str(seed))
    #log_dir = f"/home/paperspace/gym_pure/{experiment_name}/{env_name}/{noise_method}/{str(seed)}/"
    #log_dir = f"/home/ubuntu/gym/{experiment_name}/{env_name}/{noise_method}_std{str(std)}/{str(seed)}/"
    log_dir = f"/home/ubuntu/gym/{experiment_name}/{env_name}/{noise_method}/{str(seed)}/"
    os.makedirs(log_dir, exist_ok=True)

    env = make_atari(env_name)
    env = Monitor(env, log_dir, allow_early_resets=True)
    #env = DummyVecEnv([lambda: env])

    if noise_method == 'basePSN':
        param_noise = True
        policy = partial(policy, dueling=False)

    if noise_method == 'duelingPSN':
        param_noise = True

    if noise_method == 'duelingASN':
        param_noise = False

    if noise_method == 'duelingPSNASN':
        param_noise = True
        # NB!!!! This should run on a different DQN class!!!!!

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

    p = Pool(2)

    d = {
        'experiment_name': ['dqn_e15'],
        'env_name': [ 'SpaceInvaders', 'Tutankham'],
        # 'env_name': [ 'Tutankham', 'BeamRider', 'MontezumaRevenge', 'SpaceInvaders', 'Pong'],
        # 'env_name': [
        #     'Breakout', 'Tutankham', 'Enduro', 'BeamRider',
        #     'MontezumaRevenge', 'SpaceInvaders', 'Pong'
        # ],
        #'noise_method': ['basePSN', 'duelingPSN', 'duelingASN'],
        'noise_method': ['duelingASN'],
        #'noise_method': ['duelingPSNASN'],
        #'seed': [123, 456, 789],
        'seed': [789],
        # 'seed': [123],
        'total_timesteps': [100_000],
        'policy': [LnCnnPolicy],

    }

    iterator = p.imap(run_experiment, dict_product(d))

    for res in iterator:
        print('done with:')
        print(res)
        with open("/home/ubuntu/gym/dqn_experiments.log", "a") as myfile:
            t = datetime.now()
            myfile.write('done: at {} \t with {}\n'.format(t, res))




    # from stable_baselines.common.atari_wrappers import make_atari
    # from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy, LnCnnPolicy
    # from stable_baselines import DQN

    # log_dir = f"/home/paperspace/test/"
    # #log_dir = f"/home/ubuntu/gym/{experiment_name}/{env_name}/{noise_method}_std{str(std)}/{str(seed)}/"
    # os.makedirs(log_dir, exist_ok=True)

    # env = make_atari('BreakoutNoFrameskip-v4')
    # env = Monitor(env, log_dir, allow_early_resets=True)

    # model = DQN(LnCnnPolicy, env, param_noise=True, verbose=1)
    # model.learn(total_timesteps=25000)
    # model.save("deepq_breakout")