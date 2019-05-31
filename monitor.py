import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    return False


def movingAverage(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = movingAverage(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

if __name__ == '__main__':
    


total_timesteps = 10_000
env_names = ['LunarLanderContinuous-v2', 'BipedalWalker-v2']
noise_metods = ['asn', 'psn']
seeds = [123, 456, 789]


for env_name in env_names:
    for noise_metod in noise_metods:
        for seed in seeds:

            # Create log dir
            log_dir = f"/tmp/gym/{env_name}/{noise_metod}/{str(seed)}"
            os.makedirs(log_dir, exist_ok=True)

            # Create and wrap the environment
            env = gym.make(env_name)
            # Logs will be saved in log_dir/monitor.csv
            env = Monitor(env, log_dir, allow_early_resets=True)
            env = DummyVecEnv([lambda: env])

            action_noise = None
            param_noise = None
            # Add some param noise for exploration
            if noise_metod == 'asn':
                n_actions = env.action_space.shape[-1]
                action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.2) * np.ones(n_actions))
            if noise_metod == 'psn':
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.2, desired_action_stddev=0.2)

            model = DDPG(MlpPolicy, env, param_noise=param_noise, action_noise=action_noise, memory_limit=int(1e6), verbose=0)
            # Train the agent
            model.learn(total_timesteps=total_timesteps, callback=callback, seed=seed)
