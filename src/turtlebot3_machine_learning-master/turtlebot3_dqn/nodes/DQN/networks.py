import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import os
from tensorflow.keras import Model


class QNetwork(Model):
    def __init__(self, fc1_dims, fc2_dims, n_actions, name, save_directory = 'model_weights/dqn/'):
        super(QNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.save_directory = os.path.join(save_directory, self.model_name + '_dqn')
        
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc1_dims, activation='relu')
        self.fc3 = Dense(self.n_actions, activation='linear')

    def call(self, state):
        output = self.fc1(state)
        output = self.fc2(output)
        output = self.fc3(output)

        return output

class CNNQNetwork(Model):
    def __init__(self, conv1_dims, conv2_dims, fc1_dims, fc2_dims, n_actions, name, save_directory = 'model_weights/dqn/'):
        super(CNNQNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.save_directory = os.path.join(save_directory, self.model_name + '_dqn')
        
        self.conv1 = Conv2D(conv1_dims[0], conv1_dims[1], activation='relu', padding='same')
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        self.conv2 = Conv2D(conv2_dims[0], conv2_dims[1], activation='relu', padding='same')
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc1_dims, activation='relu')
        self.fc3 = Dense(self.n_actions, activation='linear')

    def call(self, state):
        state = tf.expand_dims(state, axis=0)
        x = self.conv1(state)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x