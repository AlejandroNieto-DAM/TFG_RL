import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import os

class Memory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_data(self):
        indices_muestras = np.arange(len(self.states))
        np.random.shuffle(indices_muestras)
        num_batches = int(len(self.states) / self.batch_size)
        indices_batches = []
        
        for i_batch in range(num_batches):
            indices_batches.append(indices_muestras[self.batch_size * i_batch : self.batch_size * i_batch + self.batch_size])

        if len(self.states) % self.batch_size != 0:     
            indices_batches.append(indices_muestras[self.batch_size * (num_batches) : ])

        return np.array(self.states), np.array(self.probs), np.array(self.vals), np.array(self.actions), np.array(self.rewards), np.array(self.dones), indices_batches

    def store_data(self, states, probs, vals, actions, rewards, dones):
        self.states.append(states)
        self.probs.append(probs)
        self.vals.append(vals)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.dones.append(dones)
    
    def clear_data(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []


class PPOAgent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64,
                 n_epochs=10, chkpt_dir='models/', fc1_dims = 256, fc2_dims = 256):

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir

        self.actor = Sequential([
            Dense(fc1_dims, activation='relu'),
            Dense(fc2_dims, activation='relu'),
            Dense(n_actions, activation=None)
        ])
        self.actor.compile(optimizer=Adam(learning_rate=alpha))

        self.critic = Sequential([
            Dense(fc1_dims, activation='relu'),
            Dense(fc2_dims, activation='relu'),
            Dense(1, activation=None)
        ])
        self.critic.compile(optimizer=Adam(learning_rate=alpha))
        
        self.memory = Memory(batch_size)

    def store_transition(self, state, action, probs, vals, reward, done):
        print
        self.memory.store_data(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save(self.chkpt_dir + 'actor')
        self.critic.save(self.chkpt_dir + 'critic')

    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.chkpt_dir + 'actor')
        self.critic = keras.models.load_model(self.chkpt_dir + 'critic')

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])

        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)

        action = action.numpy()[0]
        value = value.numpy()[0]
        log_prob = log_prob.numpy()[0]

        return action, log_prob, value

    # Función para calcular los retornos descontados
    def compute_discounted_returns(rewards, gamma=0.99):
        discounted_returns = np.zeros_like(rewards, dtype=np.float32)
        running_total = 0
        for t in reversed(range(len(rewards))):
            running_total = rewards[t] + gamma * running_total
            discounted_returns[t] = running_total
        return discounted_returns

    # Función para calcular las ventajas
    def compute_advantages(discounted_returns, values, next_values, gamma=0.99, lambda_=0.95):
        deltas = discounted_returns - values
        advantages = np.zeros_like(discounted_returns, dtype=np.float32)
        running_advantage = 0
        for t in reversed(range(len(deltas))):
            running_advantage = deltas[t] + (gamma * lambda_ * running_advantage)
            advantages[t] = running_advantage
        return advantages

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            # Calcular los retornos descontados
            discounted_returns = compute_discounted_returns(reward_arr, self.gamma)
            
            # Calcular las ventajas
            advantage = compute_advantages(discounted_returns, vals_arr, self.gamma, self.gae_lambda)

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch], dtype=tf.float32)
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch], dtype=tf.float32)
                    actions = tf.convert_to_tensor(action_arr[batch], dtype=tf.int32)

                    probs = self.actor(states)
                    dist = tfp.distributions.Categorical(probs)
                    new_probs = dist.log_prob(actions)

                    critic_value = self.critic(states)
                    critic_value = tf.squeeze(critic_value, axis=1)

                    prob_ratio = tf.exp(new_probs - old_probs)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.reduce_mean(tf.minimum(weighted_probs, weighted_clipped_probs))

                    returns = advantage[batch] + vals_arr[batch]
                    critic_loss = tf.reduce_mean(tf.square(returns - critic_value))

                actor_params = self.actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                self.actor.optimizer.apply_gradients(zip(actor_grads, actor_params))

                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.critic.optimizer.apply_gradients(zip(critic_grads, critic_params))

        #self.memory.clear_memory()