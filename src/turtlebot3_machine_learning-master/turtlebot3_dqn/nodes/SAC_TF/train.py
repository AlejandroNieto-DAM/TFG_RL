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
from nodes.SAC_TF.agent import SACTf2
import cv2
import tensorflow as tf

class TrainSAC:
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

        self.agent = SACTf2(input_dims = [state_size], using_camera = self.using_camera)
        #self.agent = Agent(input_dims=[state_size], env=env, n_actions=5)
        #self.agent = SACAgent(5)

    def train(self):
        rospy.loginfo("ESTAMO EN EL BUENO")

        for e in range(self.episodes):
            
            done = False
            state = self.env.reset()

            

            score = 0
            self.timestep = 0

            while not done:

                action = self.agent.choose_action(state)
                state_, reward, done = self.env.step(action)

                #image_tensor = tf.cast(state, tf.uint8)

                # Convert the TensorFlow tensor to a NumPy array
                #image_np = image_tensor.numpy()

                
                self.agent.store_data(state, action, reward, state_, done)
                
                rospy.loginfo("Action --> " + str(action) + " Reward --> " + str(reward))


                self.n_steps += 1
                if self.n_steps % self.N == 0:
                    self.env.pause_simulation()
                    c1_loss, c2_loss, a_loss, alpha_loss = self.agent.learn()
                    self.env.unpause_proxy()
                #cv2.imshow('Image', state)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()



                state = state_
                score += reward
                
                self.timestep += 1
                if self.timestep >= 500:
                    rospy.loginfo("Time out!!")
                    done = True

                if done:
                    break
            
            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-25:])

            
            
            if avg_score > self.best_score:
                self.best_score = avg_score

            #if e % 10 == 0:
            print('episode', e, 'avg score %.1f' % avg_score, 'learning_steps', self.timestep)
