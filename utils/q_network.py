# utils/q_network.py

import torch
import torch.nn as nn
import torch.optim as optim
import json

class QNetwork(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_layers=[64, 32]):
        super(QNetwork, self).__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, num_actions))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
