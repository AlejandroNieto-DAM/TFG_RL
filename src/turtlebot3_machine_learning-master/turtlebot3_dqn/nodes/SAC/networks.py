import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import os
from tensorflow.keras import Model

class Critic(Model):
    def __init__(self, fc1_dims, fc2_dims, name, save_directory = 'model_weights'):
        super(Critic, self).__init__()

        self.model_name = name
        self.save_directory = os.path.join(save_directory, self.model_name + '_sac')

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc1_dims, activation='relu')
        self.q_value = Dense(1, activation=None)

    def call(self, state, action):
        output = tf.concat([state, action], axis=-1)
        output = self.fc1(output)
        output = self.fc2(output)
        q_value = self.q_value(output)

        return q_value

class Actor(Model):
    def __init__(self, fc1_dims, fc2_dims, n_actions, name, save_directory = 'model_weights'):
        super(Actor, self).__init__()

        self.model_name = name
        self.save_directory = os.path.join(save_directory, self.model_name + '_sac')

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.noise = 1e-6
        
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc1_dims, activation='relu')
        self.mean_layer = Dense(self.n_actions, activation=None)
        self.log_std_layer = Dense(self.n_actions, activation=None)

    def call(self, state):
        output = self.fc1(state)
        output = self.fc2(output)

        mean = self.mean_layer(output)
        log_std = self.log_std_layer(output)

        #log_std = tf.clip_by_value(log_std, self.noise, 1) 
        std = tf.exp(log_std)

        dist = tfp.distributions.Normal(mean, std)
        actions = dist.sample()

        action = self.n_actions * tf.tanh(actions)

        log_probs = dist.log_prob(actions)
        log_probs -= tf.math.log(1-tf.math.pow(action,2)+self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs
