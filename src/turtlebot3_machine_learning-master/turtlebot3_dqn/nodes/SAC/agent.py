from nodes.SAC.networks import Actor, Critic, CNNActor, CNNCritic
from nodes.SAC.memory import ReplayBuffer
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import rospy

class SAC():
    def __init__(self, fc1_dims = 256, fc2_dims = 256, n_actions = 5, alpha = 0.0003, gamma = 0.99, tau = 0.01, max_size = 100000, input_dims=[364], batch_size = 128, using_camera = 0):

        self.using_camera = using_camera
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.batch_size = batch_size

        self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
        self.alpha = tf.exp(self.log_alpha)
        #self.alpha = tf.Variable(0.2, dtype=tf.float32)
        self.target_entropy = -np.prod(n_actions)
        self.gamma = gamma
        self.tau = tau

        self.target_update_interval = 1
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(max_size, input_dims, n_actions, using_camera)

        if self.using_camera:
            self.policy = CNNActor(conv1_dims=(32, (3, 3)), conv2_dims=(64, (3, 3)), fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, n_actions = self.n_actions, name = "actor")  
            self.q1 = CNNCritic(conv1_dims=(32, (3, 3)), conv2_dims=(64, (3, 3)), fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "q1")
            self.q2 = CNNCritic(conv1_dims=(32, (3, 3)), conv2_dims=(64, (3, 3)), fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "q2")
            self.target_q1 = CNNCritic(conv1_dims=(32, (3, 3)), conv2_dims=(64, (3, 3)), fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "t_q1")
            self.target_q2 = CNNCritic(conv1_dims=(32, (3, 3)), conv2_dims=(64, (3, 3)), fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "t_q2")
        else:
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

        self.target_alpha_optimizer = tf.keras.optimizers.Adam(alpha)


    def store_data(self, state, action, reward, new_state, done):
        self.memory.store_data(state, action, reward, new_state, done)

    def choose_action(self, observation):
        action, _ = self.policy(tf.convert_to_tensor([observation], dtype=tf.float32))
        return self.parse_action_from_policy(action)[0]

    def parse_action_from_policy(self, actions):
        tensor = actions
        values = tensor.numpy().reshape(-1, 5)

        probabilities = tf.nn.softmax(values, axis=1).numpy()

        selected_actions = []
        for i, prob in enumerate(probabilities):
            selected_actions.append(np.random.choice(len(prob), p=prob))
            
        return selected_actions

    def save_models(self):
        self.policy.save_weights(self.policy.save_directory + ".h5")
        self.q1.save_weights(self.q1.save_directory + ".h5")
        self.q2.save_weights(self.q2.save_directory + ".h5")
        self.target_q1.save_weights(self.target_q1.save_directory + ".h5")
        self.target_q2.save_weights(self.target_q2.save_directory + ".h5")

    def load_models(self):
        self.policy.load_weights(self.policy.save_directory + ".h5")
        self.q1.load_weights(self.q1.save_directory + ".h5")
        self.q2.load_weights(self.q2.save_directory + ".h5")
        self.target_q1.load_weights(self.target_q1.save_directory + ".h5")
        self.target_q2.load_weights(self.target_q2.save_directory + ".h5")

    def update_target(self, target_weights, weights, tau):
        new_w = []
        for (a, b) in zip(target_weights, weights):
            new_w.append(tau * b + (1 - tau) * a)
        return new_w

    def learn(self):

        if self.memory.counter < self.batch_size:
            return

        state_arr, action_arr, reward_arr, new_state_arr, dones_arr = self.memory.generate_data(self.batch_size)

        states = tf.convert_to_tensor(state_arr, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state_arr, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward_arr, dtype=tf.float32)
        actions = tf.convert_to_tensor(action_arr, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones_arr, dtype=tf.float32)


        with tf.GradientTape() as tape, tf.GradientTape() as tape2:
            next_actions, next_log_probs = self.policy(states_)
            next_actions = self.parse_action_from_policy(next_actions)

            target_q1_next = self.target_q1(states_, next_actions)
            target_q2_next = self.target_q2(states_, next_actions)

            target_q_min = tf.minimum(target_q1_next, target_q2_next) - self.alpha * next_log_probs

            y = rewards + self.gamma * (1-dones) * target_q_min

            q1 = self.q1(states, actions)
            q2 = self.q2(states, actions)

            critic_1_loss = tf.keras.losses.MSE(target_q1_next, q1)
            critic_2_loss = tf.keras.losses.MSE(target_q2_next, q2)

        critic_1_grads = tape.gradient(critic_1_loss, self.q1.trainable_variables)
        critic_2_grads = tape2.gradient(critic_2_loss, self.q2.trainable_variables)

        self.q1.optimizer.apply_gradients(zip(critic_1_grads, self.q1.trainable_variables))
        self.q2.optimizer.apply_gradients(zip(critic_2_grads, self.q2.trainable_variables))

        with tf.GradientTape() as tape_act:
            new_actions, new_log_probs = self.policy(states)
            new_actions = self.parse_action_from_policy(new_actions)

            q1_policy = self.q1(states, new_actions)
            q2_policy = self.q2(states, new_actions)

            min_q_policy = tf.minimum(q1_policy, q2_policy)

            actor_loss = tf.reduce_mean(self.alpha * new_log_probs - min_q_policy)
        
        actor_grads = tape_act.gradient(actor_loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(actor_grads, self.policy.trainable_variables))
        
        with tf.GradientTape() as tape_alpha:
            _ , log_probs = self.policy(states)
            alpha_loss = tf.reduce_mean( - self.log_alpha*(log_probs + self.target_entropy))

        grads = tape_alpha.gradient(alpha_loss, [self.log_alpha])
        self.target_alpha_optimizer.apply_gradients(zip(grads, [self.log_alpha]))
        self.alpha = tf.exp(self.log_alpha)
        
        tq1_w = self.update_target(self.target_q1.variables, self.q1.variables, self.tau)
        self.target_q1.set_weights(tq1_w)
        tq2_w = self.update_target(self.target_q2.variables, self.q2.variables, self.tau)
        self.target_q2.set_weights(tq2_w)

        return critic_1_loss, critic_2_loss, actor_loss, alpha_loss



        
    