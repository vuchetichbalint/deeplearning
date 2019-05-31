import random

import numpy as np
import tensorflow as tf
import gym


env = gym.make('CartPole-v0')
state_space_size = 4
action_space_size = 1
n_actions = 2



#sess = tf_reset()

def create_model():
    #create inputs
    # shape: None -> state_size, 1 -> action_size
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, state_space_size])
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None, n_actions])
    hidden_layer1 = tf.layers.dense(inputs=input_ph, units=164, activation=tf.nn.relu)
    hidden_layer2 = tf.layers.dense(inputs=hidden_layer1, units=150, activation=tf.nn.relu)
    output_pred = tf.layers.dense(inputs=hidden_layer2, units=n_actions)  
    return input_ph, output_ph, output_pred

input_ph, output_ph, output_pred = create_model()

# create loss
mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))

# create optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

#initialize variables
# --- sess.run(tf.global_variables_initializer())
#create saver to save model variables
# --- saver = tf.train.Saver()

# training
"""
batch_size = 32
for training_step in range(10_000):

    # get input and output data
    input_batch = ... #assert np array
    output_batch = ... # assert np array
    _, mse_run = sess.run([opt, mse], feed_dict={input_ph: input_batch, output_ph: output_batch})

    # print the mse every often so
    if training_step % 1000 == 0:
        print('{0:04d}, mse: {1:.3f}'.format(training_step, mse_run))
        saver.save(sess, './model.ckpt')
"""


with tf.Session() as sess:
    #rendering = False
    epsilon = 0.5
    gamma = 0.99
    reward_sum = 0
    total_episodes = 10000
    sess.run(tf.global_variables_initializer())


    
    for episode_number in range(total_episodes):
        state = env.reset() # Initial state of the environment
        done = False
        episode_length = 0
        while done == False:
            episode_length += 1

            #if reward_sum/batch_size > 100:     #Render environment only after avg reward reaches 100
            #    rendering = True
            #if rendering:
            #    env.render()

            #We are in state S
            #Let's run our Q function on S to get Q values for all possible actions
            # --- qval = model.predict(state.reshape(1,64), batch_size=1)
            qval = sess.run(output_pred, feed_dict={input_ph: state.reshape((1,state_space_size))})
            #action = np.zeros((action_space_size))
            if (random.random() < epsilon): #choose random action
                action = np.random.randint(0,n_actions)
            else: #choose best action from Q(s,a) values
                action = (np.argmax(qval))

            #Take action, observe new state S'
            # --- new_state = makeMove(state, action)
            #Observe reward
            # --- reward = getReward(new_state)
            
            new_state, reward, done, info = env.step(action)
            #Get max_Q(S',a)
            # --- newQ = model.predict(new_state.reshape(1,64), batch_size=1)
            newQ = sess.run(
                output_pred, 
                feed_dict={input_ph: new_state.reshape(1,state_space_size)}
            )
            maxQ = np.max(newQ)
            y = np.zeros((1,n_actions))
            #import pdb; pdb.set_trace()
            y[:] = qval[:]
            if done: # terminal state
                update = reward
            else: # non-terminal state
                update = (reward + (gamma * maxQ))
            y[0][action] = update #target output

            # --- model.fit(state.reshape(1,64), y, batch_size=1, nb_epoch=1, verbose=1)
            _, mse_run, = sess.run([opt, mse],
                feed_dict={
                    input_ph: state.reshape(1, state_space_size),
                    output_ph: y
                }
            )

            state = new_state

            #print(f'MSE: {mse_run}')
        print(f'end of episode: {episode_number}, lasts for {episode_length}, epsilon: {epsilon}')

        if epsilon > 0.1:
            epsilon -= (1/(15))




