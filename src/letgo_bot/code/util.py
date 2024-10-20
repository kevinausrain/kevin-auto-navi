import torch
import torch.nn as nn
import numpy as np
from vit import VisionTransformer
import math
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_weight(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)

def add_vision_transformer(image_size, patch_size, num_classes, dim, mlp_dim, channels, depth, heads):
    return VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        mlp_dim=mlp_dim,
        channels=channels,
        depth=depth,
        heads=heads
    )

def add_convs(layers, max_layer_num, in_out_channels, kernel_size, stride):
    for i in range(max_layer_num):
        layers.append(nn.Conv2d(in_out_channels[i][0], in_out_channels[i][1], kernel_size=kernel_size, stride=stride))


def add_full_conns(layers, max_layer_num, in_out_features):
    for i in range(max_layer_num):
        layers.append(nn.Linear(in_out_features[i][0], in_out_features[i][1]))

def state_preprocess(state, device):
    if state.ndim < 4:
         return torch.FloatTensor(state).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)
    else:
         return torch.FloatTensor(state).float().permute(0, 3, 1, 2).to(device)


def calculate_beta(cur_x, cur_y, goal_x, goal_y, angle):
    # Calculate the angle distance between the robots heading and heading toward the goal
    distance_x = goal_x - cur_x
    distance_y = goal_y - cur_y
    dot = distance_x * 1 + distance_y * 0
    mag1 = math.sqrt(math.pow(distance_x, 2) + math.pow(distance_y, 2))
    mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
    beta = math.acos(dot / (mag1 * mag2))

    beta2 = beta - angle
    if beta2 > np.pi:
        beta2 = -np.pi * 2 + beta2
    if beta2 < -np.pi:
        beta2 = 2 * np.pi + beta2

    return beta2

def display_move_in_rviz(goal_publisher, linear_speed_publisher, angular_speed_publisher, space_publisher, act, goal_x, goal_y):
    goal_marker_array = MarkerArray()
    goal_marker = Marker()
    goal_marker.header.frame_id = "odom"
    goal_marker.type, goal_marker.action = Marker.CYLINDER, Marker.ADD
    goal_marker.scale.x, goal_marker.scale.y, goal_marker.scale.z = 0.3, 0.3, 0.01
    goal_marker.color.a, goal_marker.color.r, goal_marker.color.g, goal_marker.color.b = 1.0, 1.0, 1.0, 1.0
    goal_marker.pose.orientation.w = 1.0
    goal_marker.pose.position.x, goal_marker.pose.position.y, goal_marker.pose.position.z = goal_x, goal_y, 0
    goal_marker_array.markers.append(goal_marker)
    goal_publisher.publish(goal_marker_array)

    linear_speed_marker_array = MarkerArray()
    linear_speed_marker = Marker()
    linear_speed_marker.header.frame_id = "odom"
    linear_speed_marker.type, linear_speed_marker.action = Marker.CUBE, Marker.ADD
    linear_speed_marker.scale.x, linear_speed_marker.scale.y, linear_speed_marker.scale.z = abs(act[0]), 0.1, 0.01
    linear_speed_marker.color.a, linear_speed_marker.color.r, linear_speed_marker.color.g, linear_speed_marker.color.b = 1.0, 1.0, 0.0, 0.0
    linear_speed_marker.pose.orientation.w = 1.0
    linear_speed_marker.pose.position.x, linear_speed_marker.pose.position.y, linear_speed_marker.pose.position.y = 5, 0, 0
    linear_speed_marker_array.markers.append(linear_speed_marker)
    linear_speed_publisher.publish(linear_speed_marker_array)

    angular_speed_marker_array = MarkerArray()
    angular_speed_marker = Marker()
    angular_speed_marker.header.frame_id = "odom"
    angular_speed_marker.type, angular_speed_marker.action = Marker.CUBE, Marker.ADD
    angular_speed_marker.scale.x, angular_speed_marker.scale.y, angular_speed_marker.scale.z = abs(act[1]), 0.1, 0.01
    angular_speed_marker.color.a, angular_speed_marker.color.r, angular_speed_marker.color.g, angular_speed_marker.color.b = 1.0, 1.0, 0.0, 0.0
    angular_speed_marker.pose.orientation.w = 1.0
    angular_speed_marker.pose.position.x, angular_speed_marker.pose.position.y, angular_speed_marker.pose.position.z = 5, 0.2, 0
    angular_speed_marker_array.markers.append(angular_speed_marker)
    angular_speed_publisher.publish(angular_speed_marker_array)


    space_marker_array = MarkerArray()
    space_marker = Marker()
    space_marker.header.frame_id = "odom"
    space_marker.type = Marker.CUBE
    space_marker.action = Marker.ADD
    space_marker.scale.x = 0.1
    space_marker.scale.y = 0.1
    space_marker.scale.z = 0.01
    space_marker.color.a = 1.0
    space_marker.color.r = 1.0
    space_marker.color.g = 0.0
    space_marker.color.b = 0.0
    space_marker.pose.orientation.w = 1.0
    space_marker.pose.position.x = 5
    space_marker.pose.position.y = 0.4
    space_marker.pose.position.z = 0

    space_marker_array.markers.append(space_marker)
    space_publisher.publish(space_marker_array)


