"""
QNetwork.py

Defines the QNetwork class, a neural network model for Q-learning. 
This model estimates Q-values for given states, helping an agent choose optimal actions.

Classes:
    - QNetwork: Neural network for Q-value estimation in reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, learning_rate=0.001):
        """
        Initializes the QNetwork with specified state and action sizes and learning rate.
        """
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        # Define fully connected layers for the network
        self.fc1 = nn.Linear(self.state_size, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 128)              # Second hidden layer
        self.fc3 = nn.Linear(128, self.action_size) # Output layer (Q-values for actions)
        
        # Optimizer for training the network
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Update the loss function to Mean Squared Error for Q-learning
        self.loss_fn = nn.MSELoss()
        
        # Set device for computations (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # Move model to the chosen device

    def forward(self, state):
        """
        Performs a forward pass through the network to compute Q-values for given state.
        """
        x = F.relu(self.fc1(state))  # Apply ReLU activation to first layer
        x = F.relu(self.fc2(x))      # Apply ReLU activation to second layer
        q_values = self.fc3(x)       # Output Q-values
        return q_values

    def predict(self, state):
        """
        Predicts Q-values for a given state without updating model parameters.

        Parameters:
        -----------
        state : np.ndarray
            Input state of shape [1, state_size].

        Returns:
        --------
        np.ndarray
            Q-values for each action.
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():  # Disable gradient computation for efficiency
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.forward(state)
        return q_values.cpu().numpy()  # Return as numpy array

    def train_on_batch(self, state, target):
        """
        Trains the network on a batch of states and target Q-values.

        Parameters:
        -----------
        state : np.ndarray
            Batch of states of shape [batch_size, state_size].
        target : np.ndarray
            Batch of target Q-values of shape [batch_size, action_size].

        Returns:
        --------
        float
            The scalar loss value.
        """
        self.train()  # Set to training mode
        state = torch.FloatTensor(state).to(self.device)
        target = torch.FloatTensor(target).to(self.device)  # Targets as FloatTensor for MSELoss
        
        # Forward pass to get predictions
        q_values = self.forward(state)
        
        # Calculate loss
        loss = self.loss_fn(q_values, target)
        
        # Backpropagation and optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()  # Return the scalar loss
