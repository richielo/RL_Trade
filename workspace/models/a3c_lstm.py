import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import norm_col_init, weights_init

class A3C_LSTM(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(A3C_LSTM, self).__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions

        # Hidden layers and LSTM cells
        self.fc_1 = nn.Linear(input_dim, 64)
        self.fc_2 = nn.Linear(64, 128)
        self.fc_3 = nn.Linear(128, 128)
        self.lstm_cells = nn.LSTMCell(128, 128)

        # Actor-Critic Linear Layers
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, num_actions)

        # Weight initialization
        self.apply(weights_init)
        self.fc_1.weight.data = norm_col_init(self.fc_1.weight.data, 0.01)
        self.fc_1.bias.data.fill_(0)
        self.fc_2.weight.data = norm_col_init(self.fc_2.weight.data, 0.01)
        self.fc_2.bias.data.fill_(0)
        self.fc_3.weight.data = norm_col_init(self.fc_3.weight.data, 0.01)
        self.fc_3.bias.data.fill_(0)
        self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm_cells.bias_ih.data.fill_(0)
        self.lstm_cells.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(self.fc_1(inputs))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        hx, cx = self.lstm_cells(x, (hx, cx))
        x = hx
        critic_output = self.critic_linear(x)
        actor_output = self.actor_linear(x)
        return critic_output, actor_output, (hx, cx)
