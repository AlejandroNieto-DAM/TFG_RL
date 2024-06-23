from nodes.SAC.networks import Actor, Critic
from nodes.SAC.memory import ReplayBuffer
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import rospy

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

        self.policy = Actor(fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, n_actions = self.n_actions, name = "actor")
        self.q1 = Critic(fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "q1")
        self.q2 = Critic(fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "q2")
        self.target_q1 = Critic(fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "t_q1")
        self.target_q2 = Critic(fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "t_q2")

        self.policy.compile(optimizer=Adam(learning_rate=alpha))
        self.q1.compile(optimizer=Adam(learning_rate=alpha))
        self.q2.compile(optimizer=Adam(learning_rate=alpha))
        self.target_q1.compile(optimizer=Adam(learning_rate=alpha))
        self.target_q2.compile(optimizer=Adam(learning_rate=alpha))

        self.update_target(self.target_q1.variables, self.q1.variables, 1)
        self.update_target(self.target_q2.variables, self.q2.variables, 1)


    def store_data(self, state, action, reward, new_state, done):
        self.memory.store_data(state, action, reward, new_state, done)

    def choose_action(self, observation):
        action, _ = self.policy(tf.convert_to_tensor([observation], dtype=tf.float32))
        mapped_action = (action[0, 0].numpy() + 1.0) * 2.0  # Scale to [0, 4]
        return int(np.round(mapped_action))

    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(tau * b + (1 - tau) * a)

    def save_models(self):
        self.policy.save_weights(self.policy.save_directory)
        self.q1.save_weights(self.q1.save_directory)
        self.q2.save_weights(self.q2.save_directory)
        self.target_q1.save_weights(self.target_q1.save_directory)
        self.target_q2.save_weights(self.target_q2.save_directory)

    def load_models(self):
        self.policy.load_weights(self.policy.save_directory)
        self.q1.load_weights(self.q1.save_directory)
        self.q2.load_weights(self.q2.save_directory)
        self.target_q1.load_weights(self.target_q1.save_directory)
        self.target_q2.load_weights(self.target_q2.save_directory)

    def learn(self):

        if self.memory.counter < self.batch_size:
            return

        state_arr, action_arr, reward_arr, new_state_arr, dones_arr = self.memory.generate_data(self.batch_size)

        states = tf.convert_to_tensor(state_arr, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state_arr, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward_arr, dtype=tf.float32)
        actions = tf.convert_to_tensor(action_arr, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones_arr, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            q1 = self.q1(states, actions)
            q2 = self.q2(states, actions)

            next_action, next_log_prob = self.policy(states_)

            target_q1_next = self.target_q1(states_, next_action)
            target_q2_next = self.target_q2(states_, next_action)

            target_q_min = tf.minimum(target_q1_next, target_q2_next) - self.alpha * tf.reduce_mean(next_log_prob)
            y = rewards + self.gamma * (1 - dones) * tf.squeeze(target_q_min)

            critic_1_loss = tf.reduce_mean((y - tf.squeeze(q1))**2)
            critic_2_loss = tf.reduce_mean((y - tf.squeeze(q2))**2)

        critic_1_grads = tape.gradient(critic_1_loss, self.q1.trainable_variables)
        critic_2_grads = tape.gradient(critic_2_loss, self.q2.trainable_variables)

        self.q1.optimizer.apply_gradients(zip(critic_1_grads, self.q1.trainable_variables))
        self.q2.optimizer.apply_gradients(zip(critic_2_grads, self.q2.trainable_variables))

        with tf.GradientTape() as tape_act:
            actions, log_probs = self.policy(states)
            q1_policy = self.q1(states, actions)
            q2_policy = self.q2(states, actions)

            min_q_policy = tf.minimum(q1_policy, q2_policy)

            actor_loss = tf.reduce_mean(self.alpha * log_probs - min_q_policy)

        actor_grads = tape_act.gradient(actor_loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(actor_grads, self.policy.trainable_variables))

        self.update_target(self.target_q1.variables, self.q1.variables, self.tau)
        self.update_target(self.target_q2.variables, self.q2.variables, self.tau)


