"""
Q-Network implementation for deep Q-learning.

This module implements a neural network architecture for Q-value approximation.
It features batch normalization and configurable hidden layers.

Classes:
    QNetwork: Neural network that maps states to Q-values for each possible action.
"""

# utils/q_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Dueling Q-Network Architecture with Batch Normalization."""
    
    def __init__(self, state_size, action_size, hidden_layers=[128, 64]):
        super(QNetwork, self).__init__()
        layers = []
        input_size = state_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, action_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        x = self.network(state)
        return x  # Return raw Q-values without activation
