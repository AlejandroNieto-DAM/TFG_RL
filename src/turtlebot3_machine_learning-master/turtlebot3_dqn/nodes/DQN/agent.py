from nodes.DQN.networks import QNetwork, CNNQNetwork
from nodes.DQN.memory import ReplayBuffer
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import rospy

class DQN():
    def __init__(self, fc1_dims = 256, fc2_dims = 256, n_actions = 5, epsilon_min = 0.01, gamma = 0.99, lr = 0.0003, epsilon = 1.0, max_size = 100000, input_dims=[364], batch_size = 64, using_camera=0):
        
        self.using_camera = using_Camera
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.batch_size = batch_size

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.gamma = gamma

        self.memory = ReplayBuffer(max_size, input_dims, n_actions, using_camera)

        if self.using_camera:
            self.target_model = CNNQNetwork(conv1_dims=(32, (3, 3)), conv2_dims=(64, (3, 3)), fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, n_actions=self.n_actions, name="target_model")
            self.model = CNNQNetwork(conv1_dims=(32, (3, 3)), conv2_dims=(64, (3, 3)), fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, n_actions=self.n_actions, name="model")
        else:
            self.model = QNetwork(fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, n_actions=self.n_actions, name="model")
            self.target_model = QNetwork(fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, n_actions=self.n_actions, name="target_model")
       
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
            
        state_arr, action_arr, reward_arr, new_state_arr, dones_arr = self.memory.generate_data(self.batch_size)

        states = tf.convert_to_tensor(state_arr, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state_arr, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward_arr, dtype=tf.float32)
        actions = tf.convert_to_tensor(action_arr, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones_arr, dtype=tf.float32)
        

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            next_q_values = self.target_model(states_)

            max_next_q_values = np.max(next_q_values, axis=1)

            y = np.where(dones, rewards, rewards + self.gamma * max_next_q_values)

            actions = actions[:, 0]
            actions = tf.cast(actions, tf.int32)

            q_values = tf.reduce_sum(q_values * tf.one_hot(actions, self.n_actions), axis=1)

            loss =  tf.keras.losses.MSE(y, q_values)
                                
        model_grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(model_grads, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.gamma