#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from .respawnGoal import Respawn
from .respawnCoins import RespawnCoin
from std_srvs.srv import Empty
import cv2
from cv_bridge import CvBridge, CvBridgeError
import os
from std_msgs.msg import String
bridge = CvBridge()

class Env():
    def __init__(self, action_size):

        self.number_total_coins = 1

        self.goal_x = 0
        self.goal_y = 0


        self.heading = 0
        self.action_size = action_size

        self.initGoal = True

        self.get_goalbox = False
        

        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)

        self.respawn_goal = Respawn()

        self.coins = []
        self.coins_distance = np.zeros(self.number_total_coins)
        self.picked_coins = np.zeros(self.number_total_coins)
        self.picked_coins_older_value = np.zeros(self.number_total_coins)
        self.init_coins = np.ones(self.number_total_coins)

        for i in range(self.number_total_coins):
            # TODO we have to avoid when spawning different coins to take the same position
            self.coins.append(RespawnCoin(i))
            
        

        #self.camera_topic = "/camera/image"
        #rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        #self._check_front_camera_rgb_image_raw_ready()


    def _check_front_camera_rgb_image_raw_ready(self):
        self.front_camera_rgb_image_raw = None
        rospy.loginfo("Waiting for " + self.camera_topic + " to be READY...")
        while self.front_camera_rgb_image_raw is None and not rospy.is_shutdown():
            try:
                self.front_camera_rgb_image_raw = rospy.wait_for_message(self.camera_topic, Image, timeout=5.0)
                rospy.loginfo("Current " + self.camera_topic + " READY=>")

            except:
                rospy.loginfo("Current " + self.camera_topic + " not ready yet, retrying for getting front_camera_rgb_image_raw")
        
    def image_callback(self, data):
        # Extract image data from message (replace with your specific logic)
        

        #self.front_camera_rgb_image_raw = data
        #rospy.loginfo("Tamos working " + os.getcwd())
        # Path donde se estan guardando home/nietoff/.ros
        self.front_camera_rgb_image_raw = data

        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(self.front_camera_rgb_image_raw, "bgr8")
        except CvBridgeError as e:
            rospy.loginfo("MIRA LA EXCEPTION -- " + e)
        else:
            cv2.imwrite("images/captured_image_{timestamp}.png".format(timestamp=rospy.Time.now()), cv2_img)

            #cv2.imshow("Mira la imagen", cv2_img) 
            cv2.waitKey(1)

    def pause_simulation(self):
        self.pause_proxy()

    def unpause_simulation(self):
        self.unpause_proxy()


    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance



    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def getState(self, scan):

        #rospy.loginfo("Entramos a getState!!")

        scan_range = []
        heading = self.heading
        min_range = 0.13
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:
            done = True

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        if current_distance < 0.2:
            self.get_goalbox = True

        self._get_coins_distances()

        for i in range(self.number_total_coins):
            if self.coins_distance[i] < 0.2:
                self.picked_coins[i] = 1
        
        #rospy.loginfo("Salimos de getState!!")

        return scan_range + [heading, current_distance] + self.coins_distance.tolist(), done

    def _get_coins_distances(self):
        for i in range(self.number_total_coins):
            self.coins_distance[i] = self.coins[i].getCoinDistace(self.position.x, self.position.y)
    

    def setReward(self, state, done, action):
        yaw_reward = []
        current_distance = state[-1]
        heading = state[-2]

        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.goal_distance)
        reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate)

        if done:
            rospy.loginfo("Collision!!")
            reward = -200
            self.pub_cmd_vel.publish(Twist())

        #rospy.loginfo("MIRA A VER CUANTOS HAY " + str(self.picked_coins[0]) + " " + str(self.picked_coins_older_value[0]))
        for i in range(self.number_total_coins):
            # With this if we want to avoid constantly saying that we picked a coin
            # when we did that in another step but it keeps saying that we picked it
            if self.picked_coins[i] == 1 and self.picked_coins_older_value[i] == 0:
                rospy.loginfo("\n\n\n\nCoin!!\n\n\n")
                reward = 50
                self.picked_coins_older_value[i] = 1
                self.coins[i].deleteModel()
                self.pub_cmd_vel.publish(Twist())
        
        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 200
            # With +1 we want to make sure if the robot didnt pick any coin
            # the reward to be 0
            reward *= np.array(self.picked_coins).sum() + 1
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward

    def step(self, action):

        #rospy.loginfo("NTRAMOS  del step?")

        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward = self.setReward(state, done, action)

        #rospy.loginfo("Salimos del step?")

        return np.asarray(state), reward, done

    def reset(self):
        # Wait for the threads of the initialized coins
        for i in range(self.number_total_coins):
            if self.init_coins[i] == 0:
                self.coins[i].stop_spin_move_thread()

        #rospy.loginfo("Entramos a reset!!")
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        for i in range(self.number_total_coins):
            if self.init_coins[i] == 1:
                self.coins[i].getPosition(True, False)
                self.coins[i].start_spin_move_thread()
                self.init_coins[i] = 0

        self.goal_distance = self.getGoalDistace()

        self._get_coins_distances()
        # We need to reset also if we picked or not the coins, when we do a reset
        # its clear that we didnt pick any coin 
        self.picked_coins = np.zeros(self.number_total_coins)
        self.picked_coins_older_value = np.zeros(self.number_total_coins)

        state, done = self.getState(data)
        
        #rospy.loginfo("Salimos a reset!!")


        return np.asarray(state)