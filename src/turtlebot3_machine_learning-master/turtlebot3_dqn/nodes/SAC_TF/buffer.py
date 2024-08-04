import numpy as np
import tensorflow as tf
import cv2
import os

class ReplayBuffer():
    def __init__(self, batch_size, using_camera, shape):
        
        self.using_camera = using_camera
        self.batch_size = batch_size

        self.mem_size = 3500
        self.counter = 0
    
        self.states = np.zeros((self.mem_size, *shape))
        self.new_states =np.zeros((self.mem_size, *shape))
        self.actions = np.zeros((self.mem_size))
        self.rewards = np.zeros((self.mem_size))
        self.dones = np.zeros((self.mem_size))


    def get_data(self):

        states = tf.convert_to_tensor(self.states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(self.new_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(self.actions, dtype=tf.int32)
        dones = tf.convert_to_tensor(self.dones, dtype=tf.float32)

        return states, actions, rewards, next_states,  dones
    
    def generate_batches(self):
        max_mem = min(self.counter, self.mem_size)

        return np.random.choice(max_mem, self.batch_size)


    def store_data(self, state, action, reward, new_state, done):
        
        index = self.counter % self.mem_size

        self.states[index] = state
        self.new_states[index] = new_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done

        self.counter += 1

        
    def clear_data(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
        self.dones = []