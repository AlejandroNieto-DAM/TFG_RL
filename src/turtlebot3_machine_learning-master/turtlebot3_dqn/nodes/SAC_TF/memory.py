import numpy as np
import tensorflow as tf
import cv2
import os

class ReplayBuffer():
    def __init__(self, batch_size, using_camera):
        
        self.using_camera = using_camera
        self.batch_size = batch_size

        self.states = []
        self.new_states = []
        self.actions = []
        self.rewards = []
        self.dones = []


    def get_data(self):

        states = tf.convert_to_tensor(self.states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(self.new_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(self.actions, dtype=tf.int32)
        dones = tf.convert_to_tensor(self.dones, dtype=tf.float32)

        return states, actions, rewards, next_states,  dones
    
    def generate_batches(self):
        indices_muestras = np.arange(len(self.states))
        np.random.shuffle(indices_muestras)
        num_batches = int(len(self.states) / self.batch_size)
        indices_batches = []
        
        for i_batch in range(num_batches):
            indices_batches.append(indices_muestras[self.batch_size * i_batch : self.batch_size * i_batch + self.batch_size])

        if len(self.states) % self.batch_size != 0:     
            indices_batches.append(indices_muestras[self.batch_size * (num_batches) : ])

        return indices_batches


    def store_data(self, state, action, reward, new_state, done):

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.new_states.append(new_state)
        self.dones.append(done)
        
    def clear_data(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
        self.dones = []

