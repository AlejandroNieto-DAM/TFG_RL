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
import sys
import rospy
import time 
import tensorflow as tf
import cv2
import torch
from torchvision import transforms
import math

bridge = CvBridge()

class Env():
    def __init__(self, action_size, using_camera, coins_to_spawn, laser_size):

        self.number_total_coins = coins_to_spawn
        self.using_camera = using_camera
        self.laser_size = laser_size

        self.heading = 0
        self.action_size = action_size      

        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)

        self.goal_x = 0
        self.goal_y = 0
        self.initGoal = True
        self.get_goalbox = False
        self.respawn_goal = Respawn()
        self.orientation = 0

        self._put_coins()
        
        if self.using_camera:
            self.camera_topic = "/camera/image"
            rospy.Subscriber(self.camera_topic, Image, self.image_callback)
            self._check_front_camera_rgb_image_raw_ready()

    def _put_coins(self):
        self.coins = []
        self.coins_distance = np.zeros(self.number_total_coins)
        self.picked_coins = np.zeros(self.number_total_coins)
        self.picked_coins_older_value = np.zeros(self.number_total_coins)
        self.init_coins = np.ones(self.number_total_coins)

        for i in range(self.number_total_coins):
            # TODO we have to avoid when spawning different coins to take the same position
            self.coins.append(RespawnCoin(i))

    def _get_coins_distances(self):
        for i in range(self.number_total_coins):
            self.coins_distance[i] = self.coins[i].getCoinDistace(self.position.x, self.position.y)
    
    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        return goal_distance

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
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(data, "bgr8")
            #bgr_image = cv2_img
            #rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            rgb_image = tf.image.convert_image_dtype(cv2_img, tf.float32)
            resized_image = tf.image.resize(rgb_image, [64, 64], method=tf.image.ResizeMethod.BILINEAR)
            resized_image = resized_image / 255

        except CvBridgeError as e:
            rospy.loginfo("MIRA LA EXCEPTION -- " + e)

        self.front_camera_rgb_image_raw = resized_image.numpy() # SHape 640 x 480 x 3

    def pause_simulation(self):
        self.pause_proxy()

    def unpause_simulation(self):
        self.unpause_proxy()

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.orientation = yaw

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def getState(self, scan, using_camera):

        #rospy.logdebug("Entramos en getState")

        scan_range = []
        heading = self.heading
        min_range = 0.2
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        index_of_lasers_to_keep = np.arange(0, len(scan_range), len(scan_range)/self.laser_size, dtype=np.int64)
        new_scan_range = [scan_range[i] for i in index_of_lasers_to_keep]
        scan_range = new_scan_range


        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)
        if min_range > min(scan_range) > 0:
            done = True

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        if current_distance < 0.2:
            self.get_goalbox = True

        self._get_coins_distances()

        for i in range(self.number_total_coins):
            if self.coins_distance[i] < 0.2:
                self.picked_coins[i] = 1
        
        #rospy.logdebug("Salimos de getState")
        if using_camera:
            state = self.front_camera_rgb_image_raw
            #cv2.imwrite("/home/nietoff/tfg/src/turtlebot3_machine_learning-master/turtlebot3_dqn/images/ppo_images/image_{timestamp}.png".format(timestamp=rospy.Time.now()), self.front_camera_rgb_image_raw)
            #cv2.waitKey(1)
            return state, done
        else:
            state = scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle] + self.coins_distance.tolist()

        return np.asarray(state), done
    
    """
    def calculate_reward(self):
        if self.position is None or self.goal_x is None or self.goal_y is None:
            return None  # Esperar a que se obtengan los datos

        # Calcular el vector desde el bot hasta el objetivo
        dx = self.goal_x - self.position.x
        dy = self.goal_y - self.position.y
        distance = math.sqrt(dx**2 + dy**2)

        # Vector de dirección del bot
        bot_forward = (math.cos(self.orientation), math.sin(self.orientation))

        # Calcular el producto punto
        dot_product = dx * bot_forward[0] + dy * bot_forward[1]
        cos_theta = dot_product / distance
        cos_theta = max(-1.0, min(1.0, cos_theta))  # Asegurar que esté en [-1, 1]
        theta = math.acos(cos_theta)  # Ángulo en radianes
        theta_degrees = math.degrees(theta)  # Convertir a grados

        # Determinar si el objetivo está dentro del ángulo de 60 grados
        within_cone = 1 if theta_degrees <= 70 else 0

        # Definir una penalización si no está dentro del ángulo deseado
        penalty = 0.5

        # Calcular recompensa
        epsilon = 0.05
        if within_cone:
            reward = 5 / (distance + epsilon)  # Recompensa basada en la distancia
        else:
            reward = -5 * (distance + epsilon)  # Penalización por no estar dentro del ángulo deseado

        return reward
    
    def calculate_combined_reward(self, collision_penalty):
        # Calcular distancia
        dx = self.goal_x - self.position.x
        dy = self.goal_y - self.position.y
        distance = math.sqrt(dx**2 + dy**2)

        # Calcular ángulo relativo
        bot_forward = (math.cos(self.orientation), math.sin(self.orientation))
        dot_product = dx * bot_forward[0] + dy * bot_forward[1]
        cos_theta = dot_product / distance
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta = math.acos(cos_theta)

        theta_degrees = math.degrees(theta)  # Convertir a grados

        # Determinar si el objetivo está dentro del ángulo de 60 grados
        within_cone = 1 if theta_degrees <= 70 else 0



        # Recompensa basada en la distancia y ángulo
        reward_distance = 1 / (distance + 0.01)
        reward_angle = -0.5 * theta
        

        if within_cone:
            
            reward = 2**(reward_distance + reward_angle - collision_penalty * 0.05)

        else:
            reward = math.log2(2**(reward_distance + reward_angle - collision_penalty * 0.05))

        # Penalización por colisión


        # Recompensa combinada
        #reward = (reward_distance + reward_angle - collision_penalty * 0.05)

        return reward
    """
    def setReward(self, state, done, action):
        # TODO Podemos cambiar la heurística de la formula
        # para incentivar el estar cerca de las monedas (ahora mismo no se hace nada)
        yaw_reward = []
        current_distance = state[-3]
        heading = state[-4]

        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.goal_distance)
        reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate)
        
       
        #reward = self.calculate_combined_reward(obs_distance)
        if done:
            rospy.loginfo("Collision!!")
            reward = -200
            self.pub_cmd_vel.publish(Twist())

        for i in range(self.number_total_coins):
            # With this if we want to avoid constantly saying that we picked a coin
            # when we did that in another step but it keeps saying that we picked it
            if self.picked_coins[i] == 1 and self.picked_coins_older_value[i] == 0:
                rospy.loginfo("Coin!!")
                reward = 100
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

        state, done = self.getState(data, self.using_camera)

        if self.using_camera:
            # Para calcular la recompensa necesitamos los sensores
            # por eso busco traerme el valor de los sensores y ya ponemos el 
            # reward
            s_state, _ = self.getState(data, 0)
            reward = self.setReward(s_state, done, action)
        else:
            reward = self.setReward(state, done, action)


        return state, reward, done

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

        state, done = self.getState(data, self.using_camera)
        
        #rospy.loginfo("Salimos a reset!!"
        
        return state
