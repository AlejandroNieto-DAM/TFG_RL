import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, name, save_directory='model_weights/sac/'):
        super(Actor, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.save_directory = os.path.join(save_directory, self.model_name + '_sac')


        self.net = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(fc2_dims, n_actions)
        )

    def forward(self, state):
        x = F.softmax(self.net(state), dim=-1)
        return x
    
    def evaluate(self, state, epsilon=1e-8):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * epsilon
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities        
    
    
    def get_det_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        return action.detach().cpu()
    

class Critic(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, name, save_directory='model_weights/sac/'):
        super(Critic, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.save_directory = os.path.join(save_directory, self.model_name + '_sac')


        self.net = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(fc2_dims, n_actions)
        )

    def forward(self, state):        
        return self.net(state)
"""
# Pending to be tested
class CNNCritic(Model):
    def __init__(self, input_dims, n_actions, conv1_dims, conv2_dims, fc1_dims, fc2_dims, name, save_directory = 'model_weights/sac/'):
        super(CNNCritic, self).__init__()

        self.model_name = name
        self.save_directory = os.path.join(save_directory, self.model_name + '_sac.h5')

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims


        self.net = nn.Sequential(
            nn.Conv2D(input_dims, conv1_dims[0], conv1_dims[1], activation='relu', padding='same'),
            nn.MaxPooling2D(pool_size=(2, 2)),
            nn.Conv2D(conv2_dims[0], conv2_dims[1], activation='relu', padding='same'),
            nn.MaxPooling2D(pool_size=(2, 2)),    
            nn.Flatten(),
            nn.Linear(input_dims, fc1_dims),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(fc2_dims, n_actions)
        )


    def call(self, state, action):
        x = self.conv1(state)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        output = tf.concat([x, action], axis=1)
        output = self.fc1(output)
        output = self.fc2(output)
        q_value = self.q_value(output)

        return q_value

class CNNActor(Model):
    def __init__(self, conv1_dims, conv2_dims, fc1_dims, fc2_dims, n_actions, name, save_directory = 'model_weights/sac/'):
        super(CNNActor, self).__init__()

        self.model_name = name
        self.save_directory = os.path.join(save_directory, self.model_name + '_sac')

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = 5
        self.noise = 1e-6


        self.conv1 = Conv2D(conv1_dims[0], conv1_dims[1], activation='relu', padding='same')
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        self.conv2 = Conv2D(conv2_dims[0], conv2_dims[1], activation='relu', padding='same')
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc1_dims, activation='relu')
        self.mean_layer = Dense(self.n_actions, activation=None)
        self.log_std_layer = Dense(self.n_actions, activation=None)

    def call(self, state):
        #state = tf.image.convert_image_dtype(state, tf.float32)
        x = self.conv1(state)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        mu = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = tf.clip_by_value(log_std, self.noise, 1)
        std = tf.math.exp(log_std)
        
        dist = tfp.distributions.Normal(mu, std)
        action = tf.tanh(dist.sample())  
        
        log_pi = dist.log_prob(action)
        log_pi -= tf.reduce_sum(tf.math.log(1 - action**2 + 1e-6), axis=1, keepdims=True)

        return action, log_pi
"""