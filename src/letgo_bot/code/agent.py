import numpy as np
from cpprb import PrioritizedReplayBuffer

import torch
from torch.optim import Adam
import torch.nn.functional as F

from util import hard_update, soft_update
from util import set_seed
from value import Value
from policy import Policy


# decision-making agent
class Agent(object):
    def __init__(self, action_dim, pstate_dim, seed, network_config,
                 lr_c=1e-3, lr_a=1e-3, lr_alpha=1e-4, buffer_size=5000, tau=5e-3, gamma=0.99, alpha=0.05,
                 block=2, head=4, automatic_entropy_tuning=True):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.pstate_dim = pstate_dim
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
                                                     {"obs": {"shape": (128, 160, 4)},
                                                  "act": {"shape": action_dim},
                                                  "pobs": {"shape": pstate_dim},
                                                  "next_pobs": {"shape": pstate_dim},
                                                  "rew": {},
                                                  "next_obs": {"shape": (128, 160, 4)},
                                                  "engage": {},
                                                  "done": {}})

        # Critic(Value) Network Initialization
        self.critic = Value(network_config, self.action_dim, self.pstate_dim).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr_c)
        self.critic_target = Value(network_config, self.action_dim, self.pstate_dim).to(self.device)

        hard_update(self.critic_target, self.critic)

        # Actor(Policy) Network Initialization
        if self.automatic_entropy_tuning is True:
            self.target_entropy = - self.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=lr_alpha)

        self.policy = Policy(network_config, self.action_dim, self.pstate_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr_a)


    def action(self, istate, pstate, evaluate=False):
        if istate.ndim < 4:
            istate = torch.FloatTensor(istate).float().unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            pstate = torch.FloatTensor(pstate).float().unsqueeze(0).to(self.device)
        else:
            istate = torch.FloatTensor(istate).float().permute(0, 3, 1, 2).to(self.device)
            pstate = torch.FloatTensor(pstate).float().to(self.device)

        if evaluate is False:
            action, _, _ = self.policy.act([istate, pstate])
        else:
            _, _, action = self.policy.act([istate, pstate])
        return action.detach().squeeze(0).cpu().numpy()

    def learn(self, batch_size=64):
        # Sample a batch from memory
        data = self.replay_buffer.sample(batch_size)
        istates, pstates, actions = data['obs'], data['pobs'], data['act']
        rewards, next_istates, next_pstates, dones = data['rew'], data['next_obs'], data['next_pobs'], data['done']

        istates = torch.FloatTensor(istates).permute(0, 3, 1, 2).to(self.device)
        pstates = torch.FloatTensor(pstates).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_istates = torch.FloatTensor(next_istates).permute(0, 3, 1, 2).to(self.device)
        next_pstates = torch.FloatTensor(next_pstates).to(self.device)

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.policy.act([next_istates, next_pstates])
            qf1_next_target, qf2_next_target = self.critic_target([next_istates, next_pstates, next_state_actions])
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic([istates, pstates, actions])
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.act([istates, pstates])

        qf1_pi, qf2_pi = self.critic([istates, pstates, pi])
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()


        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

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

    def store_transition(self, s, a, ps, ps_, r, s_, engage, a_exp, d=0):
        if a is not None:
            self.replay_buffer.add(obs=s,
                                   act=a,
                                   pobs=ps,
                                   next_pobs=ps_,
                                   rew=r,
                                   next_obs=s_,
                                   engage=engage,
                                   done=d)
        else:
            self.replay_buffer.add(obs=s,
                                   act=a_exp,
                                   pobs=ps,
                                   next_pobs=ps_,
                                   rew=r,
                                   next_obs=s_,
                                   engage=engage,
                                   done=d)

