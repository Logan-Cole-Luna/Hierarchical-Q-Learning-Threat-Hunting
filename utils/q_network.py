# utils/q_network.py

import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """Dueling Q-Network Architecture with Batch Normalization."""
    
    def __init__(self, state_size, action_size, hidden_layers=[128, 64]):
        super(QNetwork, self).__init__()
        # The shared part of the network processes the state input to extract features.
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.BatchNorm1d(hidden_layers[1]),
            nn.ReLU()
        )
        
        # Value stream
        # This stream estimates the value of the state, which is a scalar representing the expected 
        # return of being in the current state regardless of the action taken.
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Advantage stream
        # This stream estimates the advantage of each action, which represents 
        # how much better or worse an action is compared to others in a given state.
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
        
    def forward(self, state):
        x = self.feature(state)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
