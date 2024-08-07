import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import os
from tensorflow.keras import Model
import rospy
import numpy as np

class Actor(Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims, name, save_directory = 'model_weights/ppo/'):
        super(Actor, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.save_directory =os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + save_directory, self.model_name + '_ppo')

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
        super(Critic, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.save_directory = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + save_directory, self.model_name + '_ppo')
        
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc1_dims, activation='relu')
        self.fc3 = Dense(1, None)

    def call(self, state):
        output = self.fc1(state)
        output = self.fc2(output)
        output = self.fc3(output)

        return output

class CNNCritic(Model):
    def __init__(self, conv1_dims, conv2_dims, fc1_dims, fc2_dims, name, save_directory = 'model_weights/sac/'):
        super(CNNCritic, self).__init__()

        self.model_name = name
        self.save_directory = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + save_directory, self.model_name + '_ppo_cnn')

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.conv1 = Conv2D(conv1_dims[0], conv1_dims[1], activation='relu', padding='same')
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        self.conv2 = Conv2D(conv2_dims[0], conv2_dims[1], activation='relu', padding='same')
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q_value = Dense(1)

    def call(self, state):
        x = self.conv1(state)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        q_value = self.q_value(x)

        return q_value

class CNNActor(Model):
    def __init__(self, n_actions, conv1_dims, conv2_dims, fc1_dims, name, save_directory='model_weights/ppo/'):
        super(CNNActor, self).__init__()
        self.conv1_dims = conv1_dims
        self.conv2_dims = conv2_dims
        self.fc1_dims = fc1_dims
        self.n_actions = n_actions

        self.model_name = name
        self.save_directory = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + save_directory, self.model_name + '_ppo_cnn')

        self.conv1 = Conv2D(conv1_dims[0], conv1_dims[1], activation='relu', padding='same')
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        self.conv2 = Conv2D(conv2_dims[0], conv2_dims[1], activation='relu', padding='same')
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(n_actions, activation='softmax')

    def call(self, state):
        x = self.conv1(state)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
