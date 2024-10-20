import numpy as np
from cpprb import PrioritizedReplayBuffer

import torch
from torch.optim import Adam
import torch.nn.functional as F

from util import hard_update, soft_update, set_seed, state_preprocess
from value import Value
from policy import Policy

# decision-making agent
class Agent(object):
    def __init__(self, action_dim, goal_dim, seed, network_config,
                 lr_c=1e-3, lr_a=1e-3, lr_alpha=1e-4, buffer_size=5000, tau=5e-3, gamma=0.99, alpha=0.05,
                 block=2, head=4, automatic_entropy_tuning=True):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.goal_dim = goal_dim
        self.action_dim = action_dim

        self.itera = 0
        # set and engage_weight guidance_weight for DIL
        self.buffer_size_expert = 5000
        self.batch_expert = 0

        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.seed = int(seed)

        self.block = block
        self.head = head

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        set_seed(self.seed)

        self.replay_buffer = PrioritizedReplayBuffer(buffer_size,
                                                     {"observe": {"shape": (128, 160, 4)},
                                                  "act": {"shape": action_dim},
                                                  "goal_observe": {"shape": goal_dim},
                                                  "next_goal_observe": {"shape": goal_dim},
                                                  "reward": {},
                                                  "next_observe": {"shape": (128, 160, 4)},
                                                  "engage": {},
                                                  "done": {}})

        # Critic(Value) Network Initialization
        self.critic = Value(network_config, self.action_dim, self.goal_dim).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr_c)
        self.critic_target = Value(network_config, self.action_dim, self.goal_dim).to(self.device)

        hard_update(self.critic_target, self.critic)

        # Actor(Policy) Network Initialization
        if self.automatic_entropy_tuning is True:
            self.target_entropy = - self.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=lr_alpha)

        self.policy = Policy(network_config, self.action_dim, self.goal_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr_a)

    def action(self, current_state, goal, evaluate=False):
        if current_state.ndim < 4:
            current_state = torch.FloatTensor(current_state).float().unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            goal = torch.FloatTensor(goal).float().unsqueeze(0).to(self.device)
        else:
            current_state = torch.FloatTensor(current_state).float().permute(0, 3, 1, 2).to(self.device)
            goal = torch.FloatTensor(goal).float().to(self.device)
        if evaluate is False:
            action, _, _ = self.policy.act(current_state, goal)
        else:
            _, _, action = self.policy.act(current_state, goal)
        return action.detach().squeeze(0).cpu().numpy()

    def learn(self, batch_size=64):
        # Sample a batch from memory
        training_data = self.replay_buffer.sample(batch_size)
        current_state, goal, action = training_data['observe'], training_data['goal_observe'], training_data['act']
        reward, next_state, next_goal, done = training_data['reward'], training_data['next_observe'], training_data['next_goal_observe'], training_data['done']

        current_state = state_preprocess(current_state, self.device)
        next_state = state_preprocess(next_state, self.device)
        goal = torch.FloatTensor(goal).to(self.device)
        next_goal = torch.FloatTensor(next_goal).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)

        with torch.no_grad():
            next_action, next_state_log_prob, _ = self.policy.act(next_state, next_goal)
            qf1_next, qf2_next = self.critic_target(next_state, next_goal, next_action)
            next_q_value = reward + self.gamma * (torch.min(qf1_next, qf2_next) - self.alpha * next_state_log_prob)

        qf1, qf2 = self.critic(current_state, goal, action)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf_loss = qf1_loss + F.mse_loss(qf2, next_q_value)

        self.critic_optim.zero_grad()
        before = [param.clone() for param in self.critic.parameters()]
        qf_loss.backward()
        self.critic_optim.step()
        after = [param.clone() for param in self.critic.parameters()]
        #print('value network changed:', any([not torch.equal(b, a) for a, b in zip(before, after)]))

        action, log_prob, _ = self.policy.act(current_state, goal)

        qf1, qf2 = self.critic(current_state, goal, action)
        policy_loss = ((self.alpha * log_prob) - torch.min(qf1, qf2)).mean()


        self.policy_optim.zero_grad()
        before = [param.clone() for param in self.policy.parameters()]
        policy_loss.backward()
        self.policy_optim.step()
        after = [param.clone() for param in self.policy.parameters()]
        #print('policy network changed:', any([not torch.equal(b, a) for a, b in zip(before, after)]))

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        soft_update(self.critic_target, self.critic, self.tau)
        self.itera += 1

        return qf1_loss.item(), policy_loss.item()

    def save(self, filename, directory, reward, seed):
        torch.save(self.policy.state_dict(), '%s/%s_reward%s_seed%s_actor.pth' % (directory, filename, reward, seed))
        torch.save(self.critic.state_dict(), '%s/%s_reward%s_seed%s_critic.pth' % (directory, filename, reward, seed))
        torch.save(self.critic_target.state_dict(), '%s/%s_reward%s_seed%s_critic_target.pth' % (directory, filename, reward, seed))

    def load(self, directory, filename):
        self.policy.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.critic_target.load_state_dict(torch.load('%s/%s_critic_target.pth' % (directory, filename)))

    def buffered(self, state, action, goal, next_goal, reward, next_state, engage, action_expert, done=0):
        self.replay_buffer.add(observe=state,
                               act=action if action is not None else action_expert,
                               goal_observe=goal,
                               next_goal_observe=next_goal,
                               reward=reward,
                               next_observe=next_state,
                               engage=engage,
                               done=done)