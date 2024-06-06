from nodes.SAC.networks import Actor, Critic
from nodes.SAC.memory import ReplayBuffer
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

class SAC():
    def __init__(self, fc1_dims = 256, fc2_dims = 256, n_actions = 5, alpha = 0.0003, gamma = 0.99, tau = 0.005, max_size = 100000, input_dims=[362], batch_size = 64):

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.batch_size = batch_size

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.policy = Actor(self.fc1_dims, self.fc2_dims, self.n_actions)
        self.q1 = Critic(self.fc1_dims, self.fc2_dims)
        self.q2 = Critic(self.fc1_dims, self.fc2_dims)
        self.target_q1 = Critic(self.fc1_dims, self.fc2_dims)
        self.target_q2 = Critic(self.fc1_dims, self.fc2_dims)

        self.policy.compile(optimizer=Adam(learning_rate=alpha))
        self.q1.compile(optimizer=Adam(learning_rate=alpha))
        self.q2.compile(optimizer=Adam(learning_rate=alpha))
        self.target_q1.compile(optimizer=Adam(learning_rate=alpha))
        self.target_q2.compile(optimizer=Adam(learning_rate=alpha))

        self.update_target(self.target_q1.variables, self.q1.variables)
        self.update_target(self.target_q2.variables, self.q2.variables)


    def store_data(state, action, reward, new_state, done):
        self.memory.store_data(states, actions, rewards, new_state, done)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        action, _ = self.policy(state)
        return action[0]

    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(self.tau * b + (1 - self.tau) * a)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state_arr, action_arr, reward_arr, new_state_arr, dones_arr, batches = self.memory.generate_data()

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state_arr, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        done = tf.convert_to_tensor(dones_arr, dtype=tf.float32)

        with tf.GradientTape() as tape:
            next_action, next_log_prob = self.actor(next_state_batch)
            target_q1_next = self.target_q1(next_state_batch, next_action)
            target_q2_next = self.target_q2(next_state_batch, next_action)
            target_q_min = tf.minimum(target_q1_next, target_q2_next) - self.alpha * next_log_prob
            y = reward_batch + self.gamma * (1 - done_batch) * tf.squeeze(target_q_min, axis=1)
            
            q1 = self.critic_1(state_batch, action_batch)
            q2 = self.critic_2(state_batch, action_batch)
            
            critic_1_loss = tf.keras.losses.MSE(y, tf.squeeze(q1, axis=1))
            critic_2_loss = tf.keras.losses.MSE(y, tf.squeeze(q2, axis=1))
        
        critic_1_grads = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_grads = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_1_grads, self.critic_1.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_2_grads, self.critic_2.trainable_variables))
        
        with tf.GradientTape() as tape:
            actions, log_probs = self.actor(state_batch)
            q1 = self.critic_1(state_batch, actions)
            actor_loss = tf.reduce_mean(self.alpha * log_probs - tf.squeeze(q1, axis=1))
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        self.update_target(self.target_q1.variables, self.q1.variables)
        self.update_target(self.target_q2.variables, self.q2.variables)
