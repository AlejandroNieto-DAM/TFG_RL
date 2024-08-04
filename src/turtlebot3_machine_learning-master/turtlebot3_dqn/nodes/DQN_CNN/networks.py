import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import os
from tensorflow.keras import Model

class CustomInitializer(tf.keras.initializers.Initializer):
    def __init__(self):
        pass

    def __call__(self, shape, dtype=None):
        fan_in = shape[0]
        lim = 1. / np.sqrt(fan_in)
        return tf.random.uniform(shape, -lim, lim, dtype=dtype)

    def get_config(self):
        return {}
    
class QNetwork(Model):
    def __init__(self, fc1_dims, fc2_dims, n_actions, name, save_directory = '/model_weights/dqn/'):
        super(QNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.save_directory = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + save_directory, self.model_name + '_dqn')
        
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc1_dims, activation='relu')
        self.fc3 = Dense(self.n_actions, activation='linear')

    def call(self, state):
        output = self.fc1(state)
        output = self.fc2(output)
        output = self.fc3(output)

        return output


class CNNQNetwork(Model):
    def __init__(self, n_actions, name, save_directory = '/model_weights/dqn/'):
        super(CNNQNetwork, self).__init__()
        self.n_actions = n_actions

        self.net = Sequential([
            Conv2D(256, (8, 8), activation='relu', input_shape=(64, 64, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),
            Conv2D(256, (4, 4), activation='relu',),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),
            Flatten(),
            Dense(64),
            Dense(self.n_actions, activation='linear')
        ])

        self.model_name = name
        self.save_directory = save_directory

    def call(self, state):
        
        return self.net(state)
