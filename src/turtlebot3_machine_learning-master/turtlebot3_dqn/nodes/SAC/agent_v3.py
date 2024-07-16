import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from nodes.SAC.buffer_v2 import ReplayBuffer
from nodes.SAC.networks_v3 import Actor, QNetwork, ValueNetwork
import rospy
class SACAgent:
    def __init__(self, action_dim):
        self.actor = Actor(action_dim)
        self.q1 = QNetwork()
        self.q2 = QNetwork()
        self.value = ValueNetwork()
        self.replay_buffer = ReplayBuffer(max_size=100000)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.q_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

    

    def choose_action(self, state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.expand_dims(state, axis=0)
        logits = self.actor(state)
        action_probs = tf.nn.softmax(logits)  # Distribución de probabilidades
        action = np.random.choice(len(action_probs[0]), p=action_probs.numpy()[0])
        return action

    def learn(self):
        batch_size = 128
        # Sample from the replay buffer
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Update Q-networks
        with tf.GradientTape() as tape:
            next_action_probs = tf.nn.softmax(self.actor(next_states))
            target_value = self.value(next_states)
            target_q = rewards + (1 - dones) * 0.99 * (tf.reduce_sum(next_action_probs * target_value, axis=1))  # Valor esperado
            q1_value = self.q1(tf.concat([states, tf.one_hot(actions, depth=5)], axis=-1))
            q2_value = self.q2(tf.concat([states, tf.one_hot(actions, depth=5)], axis=-1))
            q_loss = tf.reduce_mean(tf.square(q1_value - target_q)) + tf.reduce_mean(tf.square(q2_value - target_q))

        q_grad = tape.gradient(q_loss, self.q1.trainable_variables + self.q2.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.q1.trainable_variables + self.q2.trainable_variables))
        
        # Update Value Network
        with tf.GradientTape() as tape:
            value_loss = tf.reduce_mean(tf.square(self.value(states) - tf.minimum(q1_value, q2_value)))
        
        value_grad = tape.gradient(value_loss, self.value.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grad, self.value.trainable_variables))

        # Update Actor
        with tf.GradientTape() as tape:
            logits = self.actor(states)
            action_probs = tf.nn.softmax(logits)
            action_probs = tf.reduce_sum(action_probs)
            actor_loss = -tf.reduce_mean(tf.reduce_sum(action_probs * (rewards + (1 - dones) * 0.99 * target_value), axis=1))  # Entrenamos al actor

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

