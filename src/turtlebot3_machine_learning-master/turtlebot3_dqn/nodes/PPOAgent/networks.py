import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import os
from tensorflow.keras import Model
import rospy

class Actor(Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims, name, save_directory = 'model_weights/ppo/'):
        super(Actor, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.save_directory = os.path.join(save_directory, self.model_name + '_ppo')

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(n_actions, activation='softmax')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class Critic(Model):
    def __init__(self, fc1_dims, fc2_dims, name, save_directory = 'model_weights/ppo/'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.save_directory = os.path.join(save_directory, self.model_name + '_ppo')
        
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc1_dims, activation='relu')
        self.fc3 = Dense(1, None)

    def call(self, state):
        output = self.fc1(state)
        output = self.fc2(output)
        output = self.fc3(output)

        return output


        

