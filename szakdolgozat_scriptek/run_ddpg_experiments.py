import matplotlib
matplotlib.use('Agg')

import os
from datetime import datetime

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.ddpg.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

from multiprocessing import Pool
import itertools

import tensorflow as tf




def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def callback_switch(_locals, _globals):
    n_steps = _locals['total_steps']
    total_timesteps = _locals['total_timesteps']

    if n_steps == total_timesteps//10:
        _locals['self'].param_noise = None
        _locals['self'].action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(_locals['self'].n_actions), sigma=float(0.2) * np.ones(_locals['self'].n_actions))
        # print('ok, ive switched!')
        return False
    return False

def callback_anneal_and_switch(_locals, _globals):
    n_steps = _locals['total_steps']
    total_timesteps = _locals['total_timesteps']

    if n_steps < total_timesteps//10:
        if n_steps % 1000 == 0:
            alpha = n_steps / (total_timesteps / 10)

            new_std = (1-alpha)*_locals['self'].start_std + (alpha)*_locals['self'].end_std
            _locals['self'].param_noise = AdaptiveParamNoiseSpec(initial_stddev=_locals['self'].param_noise.desired_action_stddev, desired_action_stddev=new_std)

            # print('ok, ive annealed to {}!'.format(new_std))
        return False

    if n_steps == total_timesteps//10:
        # print('ok, ive switched!')
        _locals['self'].param_noise = None
        _locals['self'].action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(_locals['self'].n_actions), sigma=float(0.2) * np.ones(_locals['self'].n_actions))
        return False

    if n_steps > total_timesteps//10:
        return False

def callback_anneal_both(_locals, _globals):
    n_steps = _locals['total_steps']
    total_timesteps = _locals['total_timesteps']

    if n_steps < total_timesteps//10:
        if n_steps % 1000 == 0:
            alpha = n_steps / (total_timesteps / 10)

            new_std = (1-alpha)*_locals['self'].start_std + (alpha)*_locals['self'].end_std
            _locals['self'].param_noise = AdaptiveParamNoiseSpec(initial_stddev=_locals['self'].param_noise.desired_action_stddev, desired_action_stddev=new_std)
            _locals['self'].action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(_locals['self'].n_actions), sigma=float(0.2-new_std) * np.ones(_locals['self'].n_actions))

            # print('ok, ive annealed to {} and {}!'.format(new_std, 0.2-new_std))
        return False

    if n_steps == total_timesteps//10:
        _locals['self'].param_noise = None
        _locals['self'].action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(_locals['self'].n_actions), sigma=float(0.2) * np.ones(_locals['self'].n_actions))
        return False

    if n_steps > total_timesteps//10:
        return False

def run_experiment(params):
    log_dir = '/home/ubuntu/gym/'
    os.makedirs(log_dir, exist_ok=True)
    with open(log_dir + "ddpg_experiments.log", "a") as myfile:
        t = datetime.now()
        myfile.write('start: at {} \t with {}\n'.format(t, params))    


    experiment_name = params['experiment_name']
    env_name = params['env_name']
    noise_method = params['noise_method']
    seed = params['seed']

    total_timesteps = params['total_timesteps']

    # Create log dir
    #log_dir = "/root/code/stable-baselines/gym/{}/{}/{}/{}/".format(experiment_name, env_name, noise_method, str(seed))
    #log_dir = f"/home/paperspace/gym/{experiment_name}/{env_name}/{noise_method}_std{str(std)}/{str(seed)}/"
    log_dir = f"/home/ubuntu/gym/{experiment_name}/{env_name}/{noise_method}/{str(seed)}/"
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    env = gym.make(env_name)
    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir, allow_early_resets=True)
    #env = DummyVecEnv([lambda: env])

    n_actions = env.action_space.shape[-1]

    other_params = {}
    if noise_method == 'base_psn':
        action_noise = None
        param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.2, desired_action_stddev=0.2)

        callback = None

    if noise_method == 'switch_psn_to_asn':
        other_params['n_actions'] = env.action_space.shape[-1]

        action_noise = None
        param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.2, desired_action_stddev=0.2)

        callback = callback_switch

    if noise_method == 'anneal_psn_and_switch_to_asn':
        other_params['n_actions'] = env.action_space.shape[-1]
        other_params['start_std'] = 0.4
        other_params['end_std'] = 0.1

        action_noise = None
        param_noise = AdaptiveParamNoiseSpec(initial_stddev=other_params['start_std'], desired_action_stddev=other_params['start_std'])

        callback = callback_anneal_and_switch

    if noise_method == 'anneal_psn_to_asn':
        other_params['n_actions'] = env.action_space.shape[-1]
        other_params['start_std'] = 0.2
        other_params['end_std'] = 0.0

        action_noise = None
        param_noise = AdaptiveParamNoiseSpec(initial_stddev=other_params['start_std'], desired_action_stddev=other_params['start_std'])

        callback = callback_anneal_both

    model = DDPG(
        LnMlpPolicy,
        env,
        param_noise=param_noise,
        action_noise=action_noise,
        memory_limit=int(1e6),
        verbose=0
    )

    # Set special params
    for k,v in other_params.items():
        setattr(model, k, v)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        seed=seed
    )

    return params

if __name__ == "__main__":

    p = Pool(36)

    d = {
        'env_name': ['MountainCarContinuous-v0', 'LunarLanderContinuous-v2', 'BipedalWalker-v2'],
        'noise_method': ['base_psn', 'switch_psn_to_asn', 'anneal_psn_and_switch_to_asn', 'anneal_psn_to_asn'],
        'seed': [123, 456, 789],
        'total_timesteps': [1000000],
        'experiment_name': ['ddpg_e14'],
        'std': [0.2],
    }

    iterator = p.imap(run_experiment, dict_product(d))

    for res in iterator:
        print('done with:')
        print(res)
        with open("/home/ubuntu/gym/ddpg_experiments.log", "a") as myfile:
            t = datetime.now()
            myfile.write('done: at {} \t with {}\n'.format(t, res))
