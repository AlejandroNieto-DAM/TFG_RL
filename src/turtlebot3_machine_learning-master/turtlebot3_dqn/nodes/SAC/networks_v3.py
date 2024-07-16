import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense 

class Actor(tf.keras.Model):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.hidden1 = Dense(256, activation='relu')
        self.hidden2 = Dense(256, activation='relu')
        self.logits = Dense(action_dim)

    def call(self, state):
        x = self.hidden1(state)
        x = self.hidden2(x)
        return self.logits(x)  # Retorna logits para la distribución categórica

class QNetwork(tf.keras.Model):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.hidden1 = Dense(256, activation='relu')
        self.hidden2 = Dense(256, activation='relu')
        self.output_layer = Dense(1)

    def call(self, state_action):
        x = self.hidden1(state_action)
        x = self.hidden2(x)
        return self.output_layer(x)

class ValueNetwork(tf.keras.Model):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.hidden1 = Dense(256, activation='relu')
        self.hidden2 = Dense(256, activation='relu')
        self.output_layer = Dense(1)

    def call(self, state):
        x = self.hidden1(state)
        x = self.hidden2(x)
        return self.output_layer(x)
