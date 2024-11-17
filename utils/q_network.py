# utils/q_network.py

import torch
import torch.nn as nn

# Base w/ batch
class QNetwork(nn.Module):
    """Neural Network for approximating Q-values with Batch Normalization."""

    def __init__(self, state_size, action_size, hidden_layers=[64, 32]):
        """
        Initialize parameters and build model with BatchNorm.

        Parameters:
        - state_size (int): Dimension of each state
        - action_size (int): Number of possible actions
        - hidden_layers (list of int): Sizes of hidden layers
        """
        super(QNetwork, self).__init__()
        layers = []
        input_size = state_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # Adding BatchNorm
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.network(state)

# Dueling  Q w/ batch
class QNetwork(nn.Module):
    """Dueling Q-Network Architecture with Batch Normalization."""
    
    def __init__(self, state_size, action_size, hidden_layers=[128, 64]):
        super(QNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_layers[0]),
            #nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            #nn.BatchNorm1d(hidden_layers[1]),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], 32),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], 32),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
        
    def forward(self, state):
        x = self.feature(state)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean())
        return q_values
