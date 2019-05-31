import random

import numpy as np
import tensorflow as tf
import gym


env = gym.make('CartPole-v0')
state_space_size = 4
action_space_size = 1
n_actions = 2


def add_noise(sess, var_names, var_shapes, noise_std=1):
    if noise_std == 0:
        return
    old_var = sess.run(var_names)
    new_var = [
        i + np.random.normal(0, noise_std, size=j)
        for i, j
        in zip(old_var, var_shapes)
    ]
    for i, j in zip(var_names, new_var):
        sess.run(i.assign(j))
    return


def create_model():
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, state_space_size])
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None, n_actions])
    hidden_layer1 = tf.layers.dense(inputs=input_ph, units=164, activation=tf.nn.relu, name='hidden_layer1')
    hidden_layer2 = tf.layers.dense(inputs=hidden_layer1, units=150, activation=tf.nn.relu, name='hidden_layer2')
    output_pred = tf.layers.dense(inputs=hidden_layer2, units=n_actions, name='output_layer')
    return input_ph, output_ph, output_pred


input_ph, output_ph, output_pred = create_model()

# create loss
mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))

# create optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# initialize variables
# --- sess.run(tf.global_variables_initializer())
# create saver to save model variables
# --- saver = tf.train.Saver()

with tf.Session() as sess:
    rendering = False
    epsilon = 0.5
    noise_std = 0.5
    gamma = 0.99
    reward_sum = 0
    total_episodes = 10000
    sess.run(tf.global_variables_initializer())
    var_names = tf.global_variables()
    trainable_var_names = [x for x in var_names if x.trainable]
    trainable_var_shapes = [i.shape for i in sess.run(var_names)]

    for episode_number in range(total_episodes):
        state = env.reset()  # Initial state of the environment
        done = False
        episode_length = 0
        while not done:
            episode_length += 1

            #if rendering:
            #    env.render()

            # We are in state S
            # Let's run our Q function on S to get Q values for all possible actions
            qval = sess.run(
                output_pred,
                feed_dict={input_ph: state.reshape((1, state_space_size))}
            )
            if (random.random() < epsilon):  # choose random action
                action = np.random.randint(0, n_actions)
            else:  # choose best action from Q(s,a) values
                action = (np.argmax(qval))

            # Take action, observe new state S' and the reward
            new_state, reward, done, info = env.step(action)
            # Get max_Q(S',a)
            newQ = sess.run(
                output_pred,
                feed_dict={input_ph: new_state.reshape(1, state_space_size)}
            )
            maxQ = np.max(newQ)
            y = np.zeros((1, n_actions))
            y[:] = qval[:]
            if done:  # terminal state
                update = reward
            else:  # non-terminal state
                update = (reward + (gamma * maxQ))
            y[0][action] = update  # target output

            _, mse_run, = sess.run(
                [opt, mse],
                feed_dict={
                    input_ph: state.reshape(1, state_space_size),
                    output_ph: y
                }
            )

            #add_noise(
            #    sess=sess,
            #   var_names=trainable_var_names
            #   var_shapes=trainable_var_shapes
            #   noise_std=noise_std/10
            #)

            state = new_state

        print(f'end of episode: {episode_number}, lasts for {episode_length}, epsilon: {epsilon}, noise_std: {noise_std}')

        if epsilon > 0.1:
            epsilon -= (1 / 15)
            noise_std = epsilon / 10
        if episode_number > 50:
            noise_std = 0
        if episode_number > 150:
            rendering = True


        #if (episode_number+10) % 50:
        #    import pdb; pdb.set_trace()
