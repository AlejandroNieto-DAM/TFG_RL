import rospy
import os
import json
import numpy as np
import random
import time
import sys
import threading

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from src.turtlebot3_dqn.environment import Env
from nodes.DQN.agent import DQN
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation

class TrainDQN:
    def __init__(self, state_size = [364], action_size = 5, N = 128, env = None, episodes = 3000, using_camera = 0):

        self.using_camera = using_camera
        self.state_size = state_size
        self.action_size = action_size
        self.N = N
        self.n_steps = 0
        self.learn_iters = 0
        self.score_history = []

        self.best_score = 200
        self.target_update = 2000

        self.episodes = episodes

        self.env = env

        self.agent = DQN(input_dims=[state_size], n_actions = action_size, using_camera = self.using_camera)
        
    def train(self):
        for e in range(self.episodes):
            done = False
            state = self.env.reset()
            score = 0
            for t in range(6000):
                action = self.agent.choose_action(state)

                state_, reward, done = self.env.step(action)

                self.n_steps += 1

                self.agent.store_data(state, action, reward, state_, done)

                if self.n_steps % N == 0:
                    self.env.pause_simulation()
                    self.agent.learn()
                    self.env.unpause_proxy()
                    self.learn_iters += 1

                state = state_
                score += reward

                rospy.loginfo("Action --> " + str(action) + " Probs --> " + str(prob) + " Reward --> " + str(reward))


                if t >= 500:
                    rospy.loginfo("Time out!!")
                    done = True


                if done:
                    self.score_history.append(score)
                    avg_score = np.mean(self.score_history[-100:])

                    if avg_score > self.best_score:
                        self.best_score = avg_score
                        self.agent.save_models()

                    break