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
        
        # Shared fully connected layers with Batch Normalization
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.BatchNorm1d(hidden_layers[1]),
            nn.ReLU()
        )
        
        # Value stream
        self.value_fc = nn.Linear(hidden_layers[1], 1)
        
        # Advantage stream
        self.advantage_fc = nn.Linear(hidden_layers[1], action_size)
    
    def forward(self, state):
        # Pass through shared layers
        x = self.shared_layers(state)
        
        # Compute value and advantage streams
        value = self.value_fc(x)  # Shape: [batch_size, 1]
        advantage = self.advantage_fc(x)  # Shape: [batch_size, action_size]
        
        # Combine value and advantage to get Q-values
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_vals  # Return Q-values
