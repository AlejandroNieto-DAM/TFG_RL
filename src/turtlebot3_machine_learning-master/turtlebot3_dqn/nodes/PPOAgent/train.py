from nodes.PPOAgent.agent import PPOAgent
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
from nodes.PPOAgent.agent import PPOAgent
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation


class TrainPPO:
    def __init__(self, state_size = [364], action_size = 5, N = 128, n_epochs = 5, batch_size = 64, alpha=0.0003, episodes = 3000, env = None, using_camera = 0):

        self.using_camera = using_camera
        self.state_size = state_size
        self.action_size = action_size
        self.N = N
        self.n_steps = 0
        self.learn_iters = 0
        self.score_history = []

        self.best_score = 200
        self.target_update = 2000

        self.batch_size = batch_size
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.episodes = episodes

        self.env = env

        self.agent = PPOAgent(n_actions = self.action_size, batch_size = self.batch_size, alpha=self.alpha,
                            n_epochs=self.n_epochs, input_dims=[self.state_size], using_camera = self.using_camera)
        # Maybe in PPo doesnt work [self.state_size] and we have to change it to self.state_size
        
    def train(self):

        rospy.loginfo("TRAINING PPO")
        
        for e in range(self.episodes):
            done = False
            state = self.env.reset()
            score = 0
            self.timestep = 0
            
            while not done:
                
                action, prob, val = self.agent.choose_action(state)
                state_, reward, done = self.env.step(action)
                self.agent.store_transition(state, prob, val, action, reward, done)

                #rospy.loginfo("Action --> " + str(action) + " Probs --> " + str(prob) + " Reward --> " + str(reward))

                
                self.n_steps += 1
                if self.n_steps % self.N == 0:
                    self.env.pause_simulation()
                    actor_loss, critic_loss = self.agent.learn()
                    #rospy.loginfo("MIRA LA LOSS DEL ACOTR " + str(actor_loss) + " MIRA LA DEL CRITIC " + str(critic_loss))
                    self.env.unpause_proxy()

                state = state_
                score += reward

                self.timestep += 1
                if self.timestep >= 500:
                    rospy.loginfo("Time out!!")
                    done = True


                if done:
                    break

            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-10:])
            
            if avg_score > self.best_score:
                self.best_score = avg_score

            #if e % 10 == 0:
            print('episode', e, 'avg score %.1f' % avg_score, 'learning_steps', self.timestep)

                