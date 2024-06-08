from nodes.DQN.networks import Q_Network
from nodes.DQN.memory import ReplayBuffer
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

class DQN():
    def __init__(self, fc1_dims = 256, fc2_dims = 256, n_actions = 5, epsilon_min = 0.01, gamma = 0.99, lr = 0.0003, epsilon = 1.0, max_size = 100000, input_dims=[362], batch_size = 64):

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.batch_size = batch_size

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.gamma = gamma

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.model = Q_Network(self.fc1_dims, self.fc2_dims, self.n_actions)
        self.target_model = Q_Network(self.fc1_dims, self.fc2_dims, self.n_actions)
       
        self.model.compile(optimizer=Adam(learning_rate=self.lr))
        self.target_model.compile(optimizer=Adam(learning_rate=self.lr))
        

    def store_data(self, states, actions, rewards, new_states, dones):
        self.memory.store_data(states, actions, rewards, new_states, dones)

    def choose_action(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.model(tf.convert_to_tensor(observation, dtype=tf.float32))
            return np.argmax(q_values)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def learn(self):
        
        if len(self.memory.states) < self.batch_size:
            return
            
        state_arr, action_arr, reward_arr, new_state_arr, dones_arr, batches = self.memory.generate_data()

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state_arr, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        done = tf.convert_to_tensor(dones_arr, dtype=tf.float32)

        with tf.GradientTape() as tape:
            next_q_values = self.target_model(states_)
            max_next_q_values = np.max(next_q_values, axis=1)

            if done:
                y = rewards
            else:
                y = rewards + self.gamma * (1 - done) * max_next_q_values

            q_values = self.model(states)
            indices = tf.range(self.batch_size)
            q_values = tf.gather_nd(q_values, tf.stack([indices, actions], axis=1))


            loss =  tf.keras.losses.MSE(y, q_values)
                                
        model_grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(model_grads, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.gamma