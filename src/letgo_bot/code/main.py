import sys

from agent import Agent

sys.path.append('/src/letgo_bot/code/')
sys.path.append('/src/letgo_bot/launch/')

import os
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
        "relu_full_conn_layer": [[32, 128], [128, 128]],
    },
    "value": {
        "conv_layer": {
            "neutron_num": [[4, 16], [16, 64], [64, 256]],
            "kernel_size": 5,
            "stride": 2
        },
        "relu_full_conn_layer1": [[290, 128], [128, 32]],
        "relu_full_conn_layer2": [[290, 128], [128, 32]],
        "full_conn_layer1": [[32, 2]],
        "full_conn_layer2": [[32, 2]],
        "embed_layer": 32
    }
}

network_config2 = {
    "policy": {
        "transformer": {"head": 4, "block": 2},
        "embed_layer": 32,
        "mean_layer": 128,
        "log_std_layer": 128,
        "relu_full_conn_layer": [[32, 128], [128, 128]],
    },
    "value": {
        "conv_layer": {
            "neutron_num": [[4, 16], [16, 64], [64, 256]],
            "kernel_size": 5,
            "stride": 2
        },
        "rule_full_conn_layer_1": [[290, 128], [128, 32]],
        "rule_full_conn_layer_2": [[290, 128], [128, 32]],
        "full_conn_layer_1": [[32, 2]],
        "full_conn_layer_2": [[32, 2]],
        "embed_layer": 32
    }
}

network_config3 = {
    "policy": {
        "transformer": {"head": 4, "block": 2},
        "embed_layer": 32,
        "mean_layer": 128,
        "log_std_layer": 128,
        "relu_full_conn_layer": [[32, 128], [128, 128]],
    },
    "value": {
        "conv_layer": {
            "neutron_num": [[4, 16], [16, 64], [64, 256]],
            "kernel_size": 5,
            "stride": 2
        },
        "rule_full_conn_layer_1": [[290, 128], [128, 32]],
        "rule_full_conn_layer_2": [[290, 128], [128, 32]],
        "full_conn_layer_1": [[32, 2]],
        "full_conn_layer_2": [[32, 2]],
        "embed_layer": 32
    }
}



def evaluate(network, eval_episodes=10, epoch=0):
    observations = deque(maxlen=4)
    env.collision = 0
    ep = 0
    avg_reward_list = []
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
                action = network.action(np.array(state), np.array(goal[:2]), evaluate=True).clip(-max_action, max_action)
                a_in = [(action[0] + 1) * linear_cmd_scale, action[1] * angular_cmd_scale]
                obs_, _, _, _, _, _, _, done, goal, target = env.step(a_in, timestep)
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
            a_in = [(act[0] + 1) * linear_cmd_scale, act[1] * angular_cmd_scale]
            obs_, _, _, _, _, _, reward, done, goal, target = env.step(a_in, count)
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
    print("\n..............................................")
    print("Average Reward over %i Evaluation Episodes, At Epoch: %i, Avg Reward: %f, Collision No.: %i" % (
    eval_episodes, epoch, reward, col))
    print("..............................................")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu

    model_name = 'navi'

    # reinforcement learning configuration
    max_steps, max_episodes, batch_size = 2, 2, 32
    actor_learn_rate, critic_learn_rate = 1e-3, 1e-3
    discount = 0.99
    soft_update_rate = 0.005
    buffer_size = 5000

    # Evaluation
    save_interval = 2
    save_threshold = 0
    eval_threshold = 1
    eval_ep = 2
    save_models = True

    auto_tune = True
    alpha = 1.0
    lr_alpha = 1e-4

    seed = 525
    robot = 'navi'
    linear_cmd_scale = 0.5
    angular_cmd_scale = 2

    mode = ""
    worlds = []
    network_configs = []

    for i in range(len(sys.argv)):
        if sys.argv[i] == "--mode":
            mode = str(sys.argv[i + 1])
        if sys.argv[i] == "--world":
            worlds = str(sys.argv[i + 1]).split("/")

    if mode == 'test':
        network_configs.append(network_config1)
        network_configs.append(network_config2)
        network_configs.append(network_config3)
    if mode == 'train':
        network_configs.append(network_config1)


    if not os.path.exists("results"):
        os.makedirs("results")
    if save_models and not os.path.exists("curves"):
        os.makedirs("curves")
    if save_models and not os.path.exists("models"):
        os.makedirs("models")

    util.set_seed(seed)

    for network_config in network_configs:
        for world in worlds:
            set_world_config(world)
            env = Environment('/home/kevin/kevin-auto-navi/src/letgo_bot/launch/main.launch', '11311')

            time.sleep(5)
            env.seed(seed)

            state, goal = env.reset()
            state_dim = state.shape
            max_action = 1

            # Initialize the agent
            agent = Agent(2, 2, seed, network_config, critic_learn_rate, actor_learn_rate, lr_alpha,
                          buffer_size, soft_update_rate, discount, alpha, block=2,
                          head=4, automatic_entropy_tuning=auto_tune)

            # Create evaluation data store
            evaluations = []

            episode = 0
            done = False
            reward_list = []
            reward_heuristic_list = []
            reward_action_list = []
            reward_freeze_list = []
            reward_target_list = []
            reward_collision_list = []
            reward_mean_list = []

            pedal_list = []
            steering_list = []

            total_timestep = 0


            # Begin the training loop
            for i in tqdm(range(0, max_episodes), ascii=True):
                episode_reward = 0
                episode_heu_reward = 0.0
                episode_act_reward = 0.0
                episode_tar_reward = 0.0
                episode_col_reward = 0.0
                episode_fr_reward = 0.0

                camera_frames = deque(maxlen=4)
                s, goal = env.reset()

                for i in range(4):
                    camera_frames.append(s)

                # current state is described by four frames taken by camera
                state = np.concatenate((camera_frames[-4], camera_frames[-3], camera_frames[-2], camera_frames[-1]), axis=-1)

                for timestep in range(max_steps):
                    if timestep == 0:
                        action = agent.action(np.array(state), np.array(goal[:2])).clip(-max_action, max_action)
                        a_in = [(action[0] + 1) * linear_cmd_scale, action[1] * angular_cmd_scale]
                        last_goal = goal
                        s_, _, _, _, _, _, reward, done, goal, target = env.step(a_in, timestep)
                        state = np.concatenate((s_, s_, s_, s_), axis=-1)

                        for i in range(4):
                            camera_frames.append(s_)

                        if done:
                            print("Bad Initialization, skip this episode.")
                            break

                        continue

                    if done or timestep == max_steps - 1:
                        episode += 1

                        done = False

                        reward_list.append(episode_reward)
                        reward_mean_list.append(np.mean(reward_list[-20:]))
                        reward_heuristic_list.append(episode_heu_reward)
                        reward_action_list.append(episode_act_reward)
                        reward_target_list.append(episode_tar_reward)
                        reward_collision_list.append(episode_col_reward)
                        reward_freeze_list.append(episode_fr_reward)

                        pedal_list.clear()
                        steering_list.clear()
                        total_timestep += timestep
                        print('Robot: ', model_name, 'Episode:', episode, 'Step:', timestep, 'Total Steps:', total_timestep,
                              'R:', episode_reward, 'Overak R:', reward_mean_list[-1], 'Expert Batch:', np.int8(agent.batch_expert), 'Temperature:', agent.alpha, '\n')

                        if episode % save_interval == 0:
                            np.save(os.path.join('curves', 'reward_seed' + str(seed) + '_' + model_name),
                                    reward_mean_list, allow_pickle=True, fix_imports=True)

                        break


                    action = agent.action(np.array(state), np.array(goal[:2])).clip(-max_action, max_action)
                    action_exp = None
                    a_in = [(action[0] + 1) * linear_cmd_scale, action[1] * angular_cmd_scale]
                    pedal_list.append(round((action[0] + 1) / 2, 2))
                    steering_list.append(round(action[1], 2))

                    last_goal = goal
                    s_, r_h, r_a, r_f, r_c, r_t, reward, done, goal, target = env.step(a_in, timestep)

                    episode_reward += reward
                    episode_heu_reward += r_h
                    episode_act_reward += r_a
                    episode_fr_reward += r_f
                    episode_col_reward += r_c
                    episode_tar_reward += r_t

                    next_state = np.concatenate((camera_frames[-3], camera_frames[-2], camera_frames[-1], s_), axis=-1)

                    # Save the tuple in replay buffer
                    agent.store_transition(state, action, last_goal[:2], goal[:2], reward, next_state, 0, action_exp,
                                           done)

                    # Train the SAC model
                    agent.learn(batch_size)

                    # Update the counters
                    state = next_state
                    camera_frames.append(s_)

            # After the training is done, evaluate the network and save it
            print('train finish, start evaluate.')
            avg_reward = evaluate(agent, eval_ep, episode)
            print('evaluate finish. avg reward is {}'.format(str(avg_reward)))
            evaluations.append(avg_reward)

            if avg_reward > save_threshold:
                agent.save(model_name, directory="models", reward=int(np.floor(avg_reward)), seed=seed)

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
