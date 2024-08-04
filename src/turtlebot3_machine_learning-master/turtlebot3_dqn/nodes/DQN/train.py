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
    def __init__(self, state_size = [364], action_size = 5, N = 128, env = None, episodes = 300, using_camera = 0):

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
        self.timestep = 0

        self.avg_scores_history = []
        self.first_train = False
        
    def train(self):
        #self.agent.load_models()
        rospy.loginfo("ESTAMOS EN DQN NORMAL")

        for e in range(self.episodes):
            done = False
            state = self.env.reset()
            score = 0
            self.timestep = 0

            rospy.loginfo("Running episode" + str(e))

            while not done:
                action = self.agent.choose_action(state)
                state_, reward, done = self.env.step(action)
                self.agent.store_data(state, action, reward, state_, done)

                self.n_steps += 1
                if self.n_steps % self.N == 0:
                    self.env.pause_simulation()
                    self.agent.learn()
                    self.env.unpause_proxy()
                    self.first_train = True

                state = state_
                score += reward

                rospy.loginfo("Action --> " + str(action) + " Reward --> " + str(reward))

                self.timestep += 1
                if self.timestep >= 500:
                    rospy.loginfo("Time out!!")
                    done = True

                if done:
                    break

            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-10:])
            
            self.avg_scores_history.append(avg_score)

            if avg_score > self.best_score:
                self.best_score = avg_score
                #self.agent.save_models()

            if self.first_train:
                self.agent.update_target_model()

            print('episode', e, 'avg score %.1f' % avg_score, 'learning_steps', self.timestep)

        with open("/ubuntutfgp2/home/nietoff/tfg/report_normal_dqn.txt", "w") as f:
            f.write(str(self.avg_scores_history))
        f.close()
