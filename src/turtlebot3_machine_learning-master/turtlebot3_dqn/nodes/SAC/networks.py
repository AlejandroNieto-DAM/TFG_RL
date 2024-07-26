import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.distributions import Categorical
import torch.nn.init as init
import rospy

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

# Pending to be tested
class CNNCritic(nn.Module):
    def __init__(self, input_dims, n_actions, conv1_dims, conv2_dims, fc1_dims, fc2_dims, name, save_directory = 'model_weights/sac/'):
        super(CNNCritic, self).__init__()

        self.model_name = name
        self.save_directory = os.path.join(save_directory, self.model_name + '_sac.h5')

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims


        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Flatten(),
            nn.Linear(12544, 256),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(256, 5)
        )

    def forward(self, state):        
        return self.net(state)


class CNNActor(nn.Module):
    def __init__(self, conv1_dims, conv2_dims, fc1_dims, fc2_dims, n_actions, name, save_directory = 'model_weights/sac/'):
        super(CNNActor, self).__init__()

        self.model_name = name
        self.save_directory = os.path.join(save_directory, self.model_name + '_sac')

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = 5
        self.noise = 1e-6

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Flatten(),
            nn.Linear(12544, 256),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(256, 5)
        )

    def forward(self, state):
        x = F.softmax(self.net(state), dim=-1)
        return x
    
    def evaluate(self, state, epsilon=1e-6):
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
        rospy.logdebug(action_probs)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.detach().cpu()