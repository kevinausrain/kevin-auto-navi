import torch
import torch.nn as nn
import util
import torch.nn.functional as F

class Value(nn.Module):
    def __init__(self, network_config, nb_actions, nb_pstate):
        super(Value, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1= nn.Conv2d(network_config['value']['conv1'][0], network_config['value']['conv1'][1],
                              network_config['value']['conv1'][2], network_config['value']['conv1'][3])
        self.conv2 = nn.Conv2d(network_config['value']['conv2'][0], network_config['value']['conv2'][1],
                               network_config['value']['conv2'][2], network_config['value']['conv2'][3])
        self.conv3 = nn.Conv2d(network_config['value']['conv3'][0], network_config['value']['conv3'][1],
                               network_config['value']['conv3'][2], network_config['value']['conv3'][3])
        self.relu_fc10 = nn.Linear(network_config['value']['relu_fc10'][0], network_config['value']['relu_fc10'][1])
        self.relu_fc11 = nn.Linear(network_config['value']['relu_fc11'][0], network_config['value']['relu_fc11'][1])
        self.relu_fc20 = nn.Linear(network_config['value']['relu_fc20'][0], network_config['value']['relu_fc20'][1])
        self.relu_fc21 = nn.Linear(network_config['value']['relu_fc21'][0], network_config['value']['relu_fc21'][1])
        self.fc1 = nn.Linear(network_config['value']['fc1'][0], network_config['value']['fc1'][1])
        self.fc2 = nn.Linear(network_config['value']['fc2'][0], network_config['value']['fc2'][1])
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc_embed = nn.Linear(nb_pstate, network_config['value']['embed_layer'])

        self.apply(util.init_weight)

    def forward(self, current_state, goal_state, next_action):
        x1 = current_state
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))

        x1 = self.avg(x1)
        x1 = x1.view(x1.size(0), -1)

        x2 = goal_state
        x2 = F.relu(self.fc_embed(x2))

        x = torch.cat([x1, x2, next_action], dim=1)
        q1, q2 = x, x


        q1 = F.relu(self.relu_fc10(q1))
        q1 = F.relu(self.relu_fc11(q1))
        q1 = self.fc1(q1)

        q2 = F.relu(self.relu_fc20(q2))
        q2 = F.relu(self.relu_fc21(q2))
        q2 = self.fc2(q2)

        return q1, q2
