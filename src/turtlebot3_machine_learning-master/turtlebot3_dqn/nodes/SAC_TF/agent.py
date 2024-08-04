from nodes.SAC_TF.networks import Actor, Critic, CNNActor, CNNCritic
from nodes.SAC_TF.memory import ReplayBuffer
from nodes.SAC_TF.buffer import ReplayBuffer as RP
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import copy


class SACTf2():
    def __init__(self, fc1_dims = 256, fc2_dims = 256, n_actions = 5, alpha = 0.0005, gamma = 0.99, tau = 0.05, max_size = 100000, input_dims=[364], batch_size = 64, using_camera = 0):

        self.using_camera = using_camera
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.batch_size = batch_size

        self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
        self.alpha = tf.exp(self.log_alpha)
        self.target_entropy = -n_actions

        self.gamma = gamma
        self.tau = tau

        #self.memory = ReplayBuffer(batch_size, using_camera)
        self.memory = RP(batch_size, using_camera, input_dims)

        if using_camera:
            self.policy = CNNActor(conv1_dims=(32, (3, 3)), conv2_dims=(64, (3, 3)), fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, n_actions = self.n_actions, name = "actor")
            self.q1 = CNNCritic(conv1_dims=(32, (3, 3)), conv2_dims=(64, (3, 3)), fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "q1", n_actions = self.n_actions)
            self.q2 = CNNCritic(conv1_dims=(32, (3, 3)), conv2_dims=(64, (3, 3)), fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "q2", n_actions = self.n_actions)
            self.target_q1 = CNNCritic(conv1_dims=(32, (3, 3)), conv2_dims=(64, (3, 3)), fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "t_q1", n_actions = self.n_actions)
            self.target_q2 = CNNCritic(conv1_dims=(32, (3, 3)), conv2_dims=(64, (3, 3)), fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "t_q2", n_actions = self.n_actions)
        else:
            self.policy = Actor(input_dims = input_dims, fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, n_actions = self.n_actions, name = "actor")
            self.q1 = Critic(input_dims = input_dims, fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "q1", n_actions = self.n_actions)
            self.q2 = Critic(input_dims = input_dims, fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "q2", n_actions = self.n_actions)
            self.target_q1 = Critic(input_dims = input_dims, fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "t_q1", n_actions = self.n_actions)
            self.target_q2 = Critic(input_dims = input_dims, fc1_dims = self.fc1_dims, fc2_dims = self.fc2_dims, name = "t_q2", n_actions = self.n_actions)
      
        self.policy_optimizer=Adam(learning_rate=alpha)
        self.q1_optimizer=Adam(learning_rate=alpha)
        self.q2_optimizer=Adam(learning_rate=alpha)
        #self.target_q1_optimizer=Adam(learning_rate=alpha)
        #self.target_q2_optimizer=Adam(learning_rate=alpha)
        self.log_alpha_optimizer = tf.keras.optimizers.Adam(alpha)

        #self.target_q1.set_weights(self.q1.get_weights())
        #self.target_q2.set_weights(self.q2.get_weights())

    def store_data(self, state, action, reward, new_state, done):
        self.memory.store_data(state, action, reward, new_state, done)

    def choose_action(self, observation):
        action = self.policy.get_det_action(tf.convert_to_tensor([observation], dtype=tf.float32))
        return action.numpy()[0]

    def update_weights(self):

        for theta_target, theta in zip(self.target_q1.trainable_variables,
                                       self.q1.trainable_variables):
            theta_target = self.tau * theta_target + (1 - self.tau) * theta

        for theta_target, theta in zip(self.target_q2.trainable_variables,
                                       self.q2.trainable_variables):
            theta_target = self.tau * theta_target + (1 - self.tau) * theta
    
        
    def learn(self):

        states, actions, rewards, next_states,  dones = self.memory.get_data()
            
        batch = self.memory.generate_batches()
   
        states_batch = tf.gather(states, batch)
        actions_batch = tf.gather(actions, batch)
        rewards_batch = tf.gather(rewards, batch)
        next_states_batch = tf.gather(next_states, batch)
        dones_batch = tf.gather(dones, batch)

        #Update critics
        with tf.GradientTape(persistent = True) as tape:
            _, action_probs, log_pis = self.policy.evaluate(next_states_batch)

            Q_target1_next = self.target_q1(next_states_batch)
            #print("Q_target1_next", Q_target1_next)
            Q_target2_next = self.target_q2(next_states_batch)
            #print("Q_target2_next", Q_target2_next)
            Q_target_next = action_probs * (tf.minimum(Q_target1_next, Q_target2_next) - self.alpha * log_pis)
            #print("Q_target_next", Q_target_next)

            Q_targets = rewards_batch + (self.gamma * (1 - dones_batch) * tf.reduce_sum(Q_target_next, axis=1))
            #print("Q_targets", Q_targets)

            q1_vals = tf.gather(self.q1(states_batch), actions_batch, batch_dims=1)
            #print("q1_vals", q1_vals)
            q2_vals = tf.gather(self.q2(states_batch), actions_batch, batch_dims=1)
            #print("q2_vals", q2_vals)

            critic_1_loss = 0.5 * keras.losses.MSE(q1_vals, Q_targets)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_vals, Q_targets)

        
        critic_1_grads = tape.gradient(critic_1_loss, self.q1.trainable_variables)
        clipped_gradients = [tf.clip_by_norm(g, 1) for g in critic_1_grads]
        self.q1_optimizer.apply_gradients(zip(clipped_gradients, self.q1.trainable_variables))

        critic_2_grads = tape.gradient(critic_2_loss, self.q2.trainable_variables)
        clipped_2_gradients = [tf.clip_by_norm(g, 1) for g in critic_2_grads]
        self.q2_optimizer.apply_gradients(zip(clipped_2_gradients, self.q2.trainable_variables))

        with tf.GradientTape() as tape2:
            action_probs = self.policy(states_batch)
            
            z = tf.cast(action_probs == 0.0, dtype=tf.float32) * 1e-8
            log_action_probabilities = tf.math.log(action_probs + z)
            
            q1_vals = self.q1(states_batch)   
            q2_vals = self.q2(states_batch)

            actor_loss = tf.reduce_mean(tf.reduce_sum((action_probs * (self.alpha * log_action_probabilities - tf.minimum(q1_vals,q2_vals) )), axis=1))

        actor_grads = tape2.gradient(actor_loss, self.policy.trainable_variables)
        #clipped_actor_gradients = [tf.clip_by_norm(g, 1) for g in actor_grads]
        self.policy_optimizer.apply_gradients(zip(actor_grads, self.policy.trainable_variables))

        #Update alpha
        with tf.GradientTape() as tape3:
            log_pis = tf.reduce_mean(tf.reduce_sum(log_pis * action_probs, axis=1))
            alpha_loss = - tf.reduce_mean(tf.exp(self.log_alpha) * (log_action_probabilities + self.target_entropy))

        alpha_grads = tape3.gradient(alpha_loss, [self.log_alpha])
        self.log_alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        self.alpha = tf.exp(self.log_alpha)

        #Update target nets
        self.update_weights()


        return critic_1_loss, critic_2_loss, actor_loss, self.alpha

     



        
    