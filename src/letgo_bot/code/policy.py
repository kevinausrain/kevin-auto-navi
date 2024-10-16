import torch
import torch.nn as nn
import util
import torch.nn.functional as F
from torch.distributions import Normal

class Policy(nn.Module):
    def __init__(self, network_config, nb_actions, nb_pstate, action_space=None):
        super(Policy, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trans = util.add_vision_transformer((128, 160), (16, 20), 2, 32, 2048, 4,
                                                 network_config['policy']['transformer']['block'], network_config['policy']['transformer']['head'])
        self.fc_embed = nn.Linear(nb_pstate, network_config['policy']['embed_layer'])
        self.relu_fc1 = nn.Linear(network_config['policy']['relu_fc1'][0], network_config['policy']['relu_fc1'][1])
        self.relu_fc2 = nn.Linear(network_config['policy']['relu_fc2'][0], network_config['policy']['relu_fc2'][1])

        self.mean_linear = nn.Linear(network_config['policy']['mean_layer'], nb_actions)
        self.log_std_linear = nn.Linear(network_config['policy']['log_std_layer'], nb_actions)

        self.apply(util.init_weight)

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, current_state, goal):
        x1, x2 = current_state, goal
        x2 = self.fc_embed(x2)
        x = self.trans.forward(x1, x2)
        x = F.relu(self.relu_fc1(x))
        x = F.relu(self.relu_fc2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        std = torch.clamp(log_std, min=-20, max=2).exp()

        return mean, std

    def act(self, current_state, goal):
        mean, std = self.forward(current_state, goal)
        normal = Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x)

        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Policy, self).to(device)
