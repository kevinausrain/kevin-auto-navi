import sys

from agent import Agent

sys.path.append('/src/letgo_bot/code/')
sys.path.append('/src/letgo_bot/launch/')

import os
from datetime import datetime
import time
import statistics
import numpy as np
from tqdm import tqdm
from collections import deque
import util
import torch

from env import Environment


network_config1 = {
    "policy": {
        "transformer": {"head": 4, "block": 2},
        "embed_layer": 32,
        "mean_layer": 128,
        "log_std_layer": 128,
        "relu_fc1": [32, 128],
        "relu_fc2": [128, 128],
    },
    "value": {
        "conv1": [4, 16, 5, 2],
        "conv2": [16, 64, 5, 2],
        "conv3": [64, 256, 5, 2],
        "relu_fc10": [290, 128],
        "relu_fc11": [128, 32],
        "relu_fc20": [290, 128],
        "relu_fc21": [128, 32],
        "fc1": [32, 2],
        "fc2": [32, 2],
        "embed_layer": 32
    }
}

def evaluate(network, network_config, world, mode, now, eval_episodes=10, epoch=0):
    observations = deque(maxlen=4)
    env.collision = 0
    ep = 0
    avg_reward_list = []
    txt = None
    if mode == 'test':
        txt = open ("test_doc/" + str(now) + "-evaluation-" + world + ".txt", "w+")
        txt.writelines(str(network_config) + "\n")

    while ep < eval_episodes:
        count = 0
        obs, goal = env.reset()
        done = False
        avg_reward = 0.0

        for i in range(4):
            observations.append(obs)

        observation = np.concatenate((observations[-4], observations[-3], observations[-2], observations[-1]), axis=-1)

        while not done and count < max_steps:

            if count == 0:
                action = network.action(np.array(initial_state), np.array(goal[:2]), evaluate=True).clip(-max_action, max_action)
                a_in = [(action[0] + 1) * linear_scalar, action[1] * angular_scalar]
                obs_, _, _, done, goal, target = env.step(a_in, timestep)
                observation = np.concatenate((obs_, obs_, obs_, obs_), axis=-1)

                for i in range(4):
                    observations.append(obs_)

                if done:
                    ep -= 1
                    env.collision -= 1
                    break

                count += 1
                continue

            act = network.action(np.array(observation), np.array(goal[:2]), evaluate=True).clip(-max_action,
                                                                                                       max_action)
            a_in = [(act[0] + 1) * linear_scalar, act[1] * angular_scalar]
            obs_, _, reward, done, goal, target = env.step(a_in, count)
            avg_reward += reward
            observation = np.concatenate((observations[-3], observations[-2], observations[-1], obs_), axis=-1)
            observations.append(obs_)
            count += 1

        ep += 1
        avg_reward_list.append(avg_reward)
        print("\n..............................................")
        print("%i Loop, Steps: %i, Avg Reward: %f, Collision No. : %i " % (ep, count, avg_reward, env.collision))
        print("..............................................")
    reward = statistics.mean(avg_reward_list)
    col = env.collision
    txt.writelines("average reward {}, over evaluation episodes {}, at epoch {}, collision {}".format(reward, eval_episodes, epoch, col))
    print("\n..............................................")
    print("Average Reward over %i Evaluation Episodes, At Epoch: %i, Avg Reward: %f, Collision No.: %i" % (
    eval_episodes, epoch, reward, col))
    print("..............................................")
    txt.close()
    return reward


def set_world_config(world):
    with open(str(os.path.abspath(os.path.dirname(__file__))).replace('/code', '/launch/world.launch'), "r+") as f:
        lines = f.readlines()

    f.close()
    lines[14] = "  <arg name=\"world_name\" value=\"$(find letgo_bot)/world/{}.world\"/>\n".format(world)

    with open(str(os.path.abspath(os.path.dirname(__file__))).replace('/code', '/launch/world.launch'), "w+") as f:
        f.writelines(lines)
    f.close()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cuda or cpu

    model_name = 'navi'

    # reinforcement learning configuration
    max_steps, max_episodes, batch_size = 500, 100, 32
    actor_learn_rate, critic_learn_rate = 1e-3, 1e-3
    discount = 0.99
    soft_update_rate = 0.005
    buffer_size = 5000

    # Evaluation
    save_interval = 50
    save_threshold = 0
    eval_threshold = 1
    eval_ep = 10
    save_models = True

    auto_tune = True
    alpha = 1.0
    lr_alpha = 1e-4

    seed = 525
    linear_scalar = 0.5
    angular_scalar = 2

    mode = ""
    worlds = []
    network_configs = []
    pth_name = None

    for i in range(len(sys.argv)):
        if sys.argv[i] == "--mode":
            mode = str(sys.argv[i + 1])
        if sys.argv[i] == "--world":
            worlds = str(sys.argv[i + 1]).split("/")
        if sys.argv[i] == "--load":
            pth_name = str(sys.argv[i + 1])

    if mode == 'test':
        network_configs.append(network_config1)
    if mode == 'train':
        network_configs.append(network_config1)



    if not os.path.exists("results"):
        os.makedirs("results")
    if save_models and not os.path.exists("curves"):
        os.makedirs("curves")
    if save_models and not os.path.exists("models"):
        os.makedirs("models")
    if save_models and not os.path.exists("test_doc"):
        os.makedirs("test_doc")

    util.set_seed(seed)

    for network_config in network_configs:
        for world in worlds:
            agent = Agent(2, 2, seed, network_config, critic_learn_rate, actor_learn_rate, lr_alpha,
                          buffer_size, soft_update_rate, discount, alpha, block=2,
                          head=4, automatic_entropy_tuning=auto_tune)

            if pth_name is not None:
                agent.load(directory="models", filename=pth_name)
                print('load success')

            set_world_config(world)
            env = Environment('/home/kevin/kevin-auto-navi/src/letgo_bot/launch/main.launch', '11311')

            time.sleep(5)

            env.seed(seed)
            initial_state, goal = env.reset()
            state_dim = initial_state.shape
            max_action = 1

            # Create evaluation data store
            evaluations = []

            episode = 0
            done = False
            reward_list = []
            reward_mean_list = []

            linear_move_list = []
            angular_move_list = []

            total_timestep = 0

            now = datetime.now()

            # record training data
            txt = None
            if mode == 'test':
                txt = open ("test_doc/" + str(now) + "-test-" + world + ".txt", "w+")
                txt.writelines(str(network_config) + "\n")

            # Begin the training loop
            for i in tqdm(range(0, max_episodes), ascii=True):
                episode_reward = 0
                camera_frames = deque(maxlen=4)
                initial_camera_frame, goal = env.reset()

                for i in range(4):
                    camera_frames.append(initial_camera_frame)

                # current state is described by four frames taken by camera
                initial_state = np.concatenate((camera_frames[-4], camera_frames[-3], camera_frames[-2], camera_frames[-1]), axis=-1)

                for timestep in range(max_steps):
                    print(timestep)
                    if timestep == 0:
                        # get action from current state based on agent's policy network
                        action = agent.action(np.array(initial_state), np.array(goal[:2])).clip(-max_action, max_action)
                        action_taken = [(action[0] + 1) * linear_scalar, action[1] * angular_scalar]
                        last_goal = goal
                        camera_frame, _, reward, done, goal, target = env.step(action_taken, timestep)
                        initial_state = np.concatenate((camera_frame, camera_frame, camera_frame, camera_frame), axis=-1)

                        for i in range(4):
                            camera_frames.append(camera_frame)

                        if done:
                            print("Bad Initialization, skip this episode.")
                            break
                        continue

                    # if this episode finish
                    if done or timestep == max_steps - 1:
                        episode += 1

                        done = False

                        reward_list.append(episode_reward)
                        reward_mean_list.append(np.mean(reward_list[-20:]))

                        linear_move_list.clear()
                        angular_move_list.clear()
                        total_timestep += timestep

                        if mode == 'test':
                            txt.writelines("test world: {}, episode: {}, total reward: {}\n".format(world, episode, episode_reward))

                        if episode % save_interval == 0:
                            np.save(os.path.join('curves', 'reward_seed' + str(seed) + '_' + model_name),
                                    reward_mean_list, allow_pickle=True, fix_imports=True)

                        break


                    action = agent.action(np.array(initial_state), np.array(goal[:2])).clip(-max_action, max_action)
                    action_exp = None
                    action_taken = [(action[0] + 1) * linear_scalar, action[1] * angular_scalar]
                    linear_move_list.append(round((action[0] + 1) / 2, 2))
                    angular_move_list.append(round(action[1], 2))

                    last_goal = goal
                    camera_frame, r_collision, reward, done, goal, target = env.step(action_taken, timestep)

                    if r_collision == -100 and timestep >= 50:
                        print("Collision")
                        done = True

                    episode_reward += reward
                    next_state = np.concatenate((camera_frames[-3], camera_frames[-2], camera_frames[-1], camera_frame), axis=-1)

                    # Save states in replay buffer
                    agent.store_transition(initial_state, action, last_goal[:2], goal[:2], reward, next_state, 0, action_exp,
                                           done)

                    # Train the SAC model
                    agent.learn(batch_size)

                    # Update the counters
                    initial_state = next_state
                    camera_frames.append(camera_frame)

                    if mode == 'test':
                        txt.writelines("test world: {}, current step: {}. step reward: {}\n".format(world, timestep, reward))

            txt.close()

            # After the training is done, evaluate the network and save it
            agent.save(str(now) + "-" + str(model_name) + "-", directory="models", reward=int(np.floor(1)), seed=seed)

            print('train finish, start evaluate.')
            avg_reward = evaluate(agent, network_config, world, mode, now, eval_ep, episode)
            print('evaluate finish. avg reward is {}'.format(str(avg_reward)))
            evaluations.append(avg_reward)
            print('avg_reward is {}, threshold is {}'.format(avg_reward, save_threshold))

            np.save(os.path.join('curves', 'reward_seed' + str(seed) + '_' + model_name), reward_mean_list,
                    allow_pickle=True, fix_imports=True)

            try:
                os.system("killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3 rviz")
            except Exception as e:
                pass

            print('last round stops, new rounds will start in 10 seconds if there is')
            time.sleep(10)



    try:
        os.system("killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3 rviz")
    except Exception as e:
        pass

    print('process end')
