import os
from os import path

import time
import math
import random
import numpy as np
from numpy import inf
from collections import deque
from squaternion import Quaternion

import cv2
from cv_bridge import CvBridge

import rospy
import subprocess
import util
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import sensor_msgs.point_cloud2 as pc2
from sympy.physics.units import current
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import Image, LaserScan, PointCloud2

from gazebo_msgs.msg import ModelState


x_low, x_high, y_low, y_high = 0, 1, 2, 3
pos_coord = [[-6.7, -6.5, -3.4, 3.6],
             [3.5, 5.2, -3.4, 3.6],
             [-6.4, 3.5, -3.4, -2.8],
             [-6.4, 3.5, 3.1, 3.6],
             [-1.8, -1.4, -3.4, 3.6]]


# Check if the random goal position is located on an obstacle
def check_goal_on_obstacle(x, y):
    for pos in pos_coord:
        if pos[x_low] < x < pos[x_high] and pos[y_low] < y < pos[y_high]:
            return True

    return False

class Environment:
    def __init__(self, launchfile, master_uri):
        self.br = CvBridge()

        self.robot_x, self.robot_y = 0, 0
        self.goal_x, self.goal_y = 1, 1

        self.velodyne_data = np.ones(20) * 10
        self.current_laser, self.current_odom, self.current_image_frame = None, None, None

        self.collision = 0
        self.last_act = 0

        self.x_pos_list, self.y_pos_list = deque(maxlen=5), deque(maxlen=5)

        self.agent_state = ModelState()
        self.agent_state.model_name = 'navi'
        self.agent_state.pose.position.x, self.agent_state.pose.position.y, self.agent_state.pose.position.z = 0.0, 0.0, 0.0
        (self.agent_state.pose.orientation.x, self.agent_state.pose.orientation.y,
         self.agent_state.pose.orientation.z, self.agent_state.pose.orientation.w) = 0.0, 0.0, 0.0, 0.0

        self.distance = math.sqrt(math.pow(self.robot_x - self.goal_x, 2) + math.pow(self.robot_y - self.goal_y, 2))
        self.gaps = [[-1.6, -np.pi + np.pi / 20]]

        for m in range(19):
            self.gaps.append([self.gaps[m][1], self.gaps[m][1] + np.pi / 20])
        self.gaps[-1][-1] += 0.03

        try:
            subprocess.Popen(["roscore", "-p", master_uri])
        except OSError as e:
            raise e

        # Launch the simulation with gym initialization
        rospy.init_node('agent', anonymous=True)

        if not path.exists(launchfile):
            raise IOError("File " + launchfile + " does not exist")

        time.sleep(10)

        subprocess.Popen(["roslaunch", "-p", master_uri, launchfile])

        # publish action
        self.vel_pub = rospy.Publisher('/navi/cmd_vel', Twist, queue_size=1)
        self.set_state = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        topic1 = "vis_mark_array"
        self.goal_publisher = rospy.Publisher(topic1, MarkerArray, queue_size=3)
        self.linear_speed_publisher = rospy.Publisher('linear_marker_array', MarkerArray, queue_size=1)
        self.angular_speed_publisher = rospy.Publisher('angular_marker_array', MarkerArray, queue_size=1)
        self.publisher4 = rospy.Publisher('vis_mark_array4', MarkerArray, queue_size=1)

        # receive sensor (laser/camera/pointcloud) data to observe environment
        self.velodyne = rospy.Subscriber('/velodyne_points', PointCloud2, self.velodyne_callback, queue_size=1)
        self.laser = rospy.Subscriber('/front_laser/scan', LaserScan, self.laser_callback, queue_size=1)
        self.odom = rospy.Subscriber('/navi/odom', Odometry, self.odom_callback, queue_size=1)
        self.image_fish = rospy.Subscriber('/camera/fisheye/image_raw', Image, self.image_fish_callback, queue_size=1)

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    # Read velodyne pointcloud data and turn it into distance data
    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(20) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])  # * -1
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def laser_callback(self, laser):
        self.current_laser = laser

    def odom_callback(self, odom):
        self.current_odom = odom

    def image_fish_callback(self, rgb_data):
        image = self.br.imgmsg_to_cv2(rgb_data, "mono8")
        self.current_image_frame = np.expand_dims(cv2.resize(image[80:400, 140:500], (160, 128)), axis=2)

    # Detect a collision from laser data
    def detect_collision(self, laser_data):
        min_range = 0.5
        min_laser = 2
        collision = False

        for i, item in enumerate(laser_data.ranges):
            if min_laser > laser_data.ranges[i]:
                min_laser = laser_data.ranges[i]
            if min_range > laser_data.ranges[i] > 0:
                collision = True
        return collision, min_laser

    # act and reward, act is given by policy network and reward is calculated based on various factors
    def step(self, act, timestep):
        # in env or in agent?
        vel_cmd = Twist()
        vel_cmd.linear.x = act[0]
        vel_cmd.angular.z = act[1]
        self.vel_pub.publish(vel_cmd)

        done = False
        target = False

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(0.1)

        dataOdom = None
        while dataOdom is None:
            try:
                dataOdom = rospy.wait_for_message('/navi/odom', Odometry, timeout=0.1)
            except:
                pass

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/front_laser/scan', LaserScan, timeout=0.1)
            except:
                pass

        data_obs = None

        data_obs_fish = None
        while data_obs_fish is None:
            try:
                data_obs_fish = rospy.wait_for_message('/camera/fisheye/image_raw', Image, timeout=0.1)
            except:
                pass

        time.sleep(0.1)
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")


        current_laser = self.current_laser
        current_odom = self.current_odom
        current_camera_frames = self.current_image_frame

        # cloud point
        velodyne_state = []
        velodyne_state[:] = self.velodyne_data[:]

        collision, min_laser = self.detect_collision(current_laser)

        # Calculate robot heading from odometer data
        self.robot_x, self.robot_y = current_odom.pose.pose.position.x, current_odom.pose.pose.position.y

        self.x_pos_list.append(round(self.robot_x, 2))
        self.y_pos_list.append(round(self.robot_y, 2))

        quaternion = Quaternion(
            current_odom.pose.pose.orientation.w,
            current_odom.pose.pose.orientation.x,
            current_odom.pose.pose.orientation.y,
            current_odom.pose.pose.orientation.z
        )

        # Calculate distance to the goal from the robot
        distance = math.sqrt(math.pow(self.robot_x - self.goal_x, 2) + math.pow(self.robot_y - self.goal_y, 2))
        beta2 = util.calculate_beta(self.robot_x, self.robot_y, self.goal_x, self.goal_y, round(quaternion.to_euler(degrees=False)[2], 4))

        # Publish visual data in Rviz to display
        util.display_move_in_rviz(self.goal_publisher, self.linear_speed_publisher, self.angular_speed_publisher, self.publisher4, act, self.goal_x, self.goal_y)

        # reward calculation
        reward = 0.0
        reward += (self.distance - distance) * 20
        reward += act[0] * 2 - abs(act[1])
        reward += - abs(act[1] - self.last_act) / 4

        self.distance = distance

        reward_collision = 0.0

        # Detect if the goal has been reached and give a large positive reward
        if distance < 0.2:
            target = True
            done = True
            self.distance = math.sqrt(math.pow(self.robot_x - self.goal_x, 2) + math.pow(self.robot_y - self.goal_y, 2))
            reward += 100

        # Detect if ta collision has happened and give a large negative reward
        if collision:
            self.collision += 1
            reward_collision = -100
            reward += reward_collision

        beta2 = beta2 / np.pi
        to_goal = np.array([distance, beta2, act[0], act[1]])

        state = current_camera_frames / 255
        self.last_act = act[1]
        return state, reward_collision, reward, done, to_goal, target

    def check_move_slow(self, buffer):
        it = iter(buffer)
        try:
            first = next(it)
        except StopIteration:
            return True
        return all((abs(first - x) < 0.1) for x in buffer)

    def reset(self):
        # Reset the environment and return initial observation
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0., 0., angle)
        object_state = self.agent_state

        x = 0
        y = 0
        robot_free_collision = False
        while not robot_free_collision:
            x = np.random.uniform(-7.0, 7.0)
            y = np.random.uniform(-4.0, 4.0)
            robot_free_collision = check_goal_on_obstacle(x, y)
        object_state.pose.position.x, object_state.pose.position.y = x, y
        object_state.pose.orientation.x, object_state.pose.orientation.y, object_state.pose.orientation.z, object_state.pose.orientation.w = (
            quaternion.x, quaternion.y, quaternion.z, quaternion.w)

        self.set_state.publish(object_state)
        self.robot_x, self.robot_y = object_state.pose.position.x, object_state.pose.position.y

        self.set_new_goal()
        self.distance = math.sqrt(math.pow(self.robot_x - self.goal_x, 2) + math.pow(self.robot_y - self.goal_y, 2))

        data = None
        data_obs_fish = None
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")
        while data is None:
            try:
                data = rospy.wait_for_message('/front_laser/scan', LaserScan, timeout=0.5)
            except:
                pass
        laser_state = np.array(data.ranges[:])
        laser_state[laser_state == inf] = 10

        while data_obs_fish is None:
            try:
                data_obs_fish = rospy.wait_for_message('/camera/fisheye/image_raw', Image, timeout=0.1)
            except:
                pass

        camera_image = data_obs_fish
        image = self.br.imgmsg_to_cv2(camera_image, "mono8")
        image = np.expand_dims(cv2.resize(image[80:400, 140:500], (160, 128)), axis=2)
        state = image / 255

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        distance = math.sqrt(math.pow(self.robot_x - self.goal_x, 2) + math.pow(self.robot_y - self.goal_y, 2))
        beta2 = util.calculate_beta(self.robot_x, self.robot_y, self.goal_x, self.goal_y, angle)
        beta2 = beta2 / np.pi
        goal = np.array([distance, beta2, 0.0, 0.0])
        return state, goal

    # set a new goal and check obstacle collision
    def set_new_goal(self):
        new_goal_ok = False

        while not new_goal_ok:
            self.goal_x = self.robot_x + random.uniform(10.0, -10.0)
            self.goal_y = self.robot_y + random.uniform(10.0, -10.0)

            euclidean_dist = math.sqrt((self.goal_x - self.robot_x) ** 2 + (self.goal_y - self.robot_y) ** 2)
            if euclidean_dist < 5:
                new_goal_ok = False
                continue

            new_goal_ok = check_goal_on_obstacle(self.goal_x, self.goal_y)
