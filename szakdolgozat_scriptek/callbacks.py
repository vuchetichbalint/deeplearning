import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec




def callback_switch(_locals, _globals):
    n_steps = _locals['total_steps']
    total_timesteps = _locals['total_timesteps']

    if n_steps == total_timesteps//10:
        _locals['self'].param_noise = None
        _locals['self'].action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(_locals['self'].n_actions), sigma=float(0.2) * np.ones(_locals['self'].n_actions))
        print('ok, ive switched!')
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

            print('ok, ive annealed to {}!'.format(new_std))
        return False

    if n_steps == total_timesteps//10:
        print('ok, ive switched!')
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

            print('ok, ive annealed to {} and {}!'.format(new_std, 0.2-new_std))
        return False

    if n_steps == total_timesteps//10:
        _locals['self'].param_noise = None
        _locals['self'].action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(_locals['self'].n_actions), sigma=float(0.2) * np.ones(_locals['self'].n_actions))
        return False

    if n_steps > total_timesteps//10:
        return False



# Create log dir
log_dir = "/home/ubuntu/workspace/logs/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make('LunarLanderContinuous-v2')
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

# Add some param noise for exploration
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.2, desired_action_stddev=0.2)
model = DDPG('MlpPolicy', env, param_noise=param_noise, memory_limit=int(1e6), verbose=0)
model.n_actions = env.action_space.shape[-1]
model.start_std = 0.2
model.end_std = 0.0
x = callback_anneal_both
# Train the agent
model.learn(total_timesteps=200000, callback=None)



