import torch
import torch.nn as nn
import util
import torch.nn.functional as F

class Value(nn.Module):
    def __init__(self, network_config, nb_actions, nb_pstate):
        super(Value, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.relu_convs = nn.ModuleList()
        self.relu_fc1s = nn.ModuleList()
        self.relu_fc2s = nn.ModuleList()
        self.fc1s = nn.ModuleList()
        self.fc2s = nn.ModuleList()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc_embed = nn.Linear(nb_pstate, network_config['value']['embed_layer'])

        util.add_convs(self.relu_convs, len(network_config['value']['conv_layer']['neutron_num']), network_config['value']['conv_layer']['neutron_num'],
                                         network_config['value']['conv_layer']['kernel_size'], network_config['value']['conv_layer']['stride'])

        util.add_full_conns(self.relu_fc1s, len(network_config['value']['relu_full_conn_layer1']), network_config['value']['relu_full_conn_layer1'])
        util.add_full_conns(self.fc1s, len(network_config['value']['full_conn_layer1']), network_config['value']['full_conn_layer1'])

        util.add_full_conns(self.relu_fc2s, len(network_config['value']['relu_full_conn_layer2']), network_config['value']['relu_full_conn_layer2'])
        util.add_full_conns(self.fc2s, len(network_config['value']['full_conn_layer2']), network_config['value']['full_conn_layer2'])

        self.apply(util.init_weight)

    def forward(self, cur_state, goal_state, next_action):
        x1 = cur_state
        x1 = x1.to(self.device)
        for conv in self.relu_convs:
            x1 = F.relu(conv(x1))

        x1 = self.avg(x1)
        x1 = x1.view(x1.size(0), -1)

        x2 = goal_state
        x2 = F.relu(self.fc_embed(x2))

        x = torch.cat([x1, x2, next_action], dim=1)
        q1, q2 = x, x

        for relu_fc in self.relu_fc1s:
            q1 = F.relu(relu_fc(q1))
        for fc in self.fc1s:
            q1 = fc(q1)
        for relu_fc in self.relu_fc2s:
            q2 = F.relu(relu_fc(q2))
        for fc in self.fc2s:
            q2 = fc(q2)

        return q1, q2
