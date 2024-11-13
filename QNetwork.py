# QNetwork.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, learning_rate=0.001, device=None):
        """
        Initializes the Q-Network.

        Parameters:
        - state_size (int): Dimension of each state
        - action_size (int): Dimension of each action
        - learning_rate (float): Learning rate for the optimizer
        - device (torch.device): Device to run the model on (CPU or GPU)
        """
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        # Define the network architecture
        self.fc1 = nn.Linear(self.state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.action_size)

        # Initialize weights (optional but recommended)
        self._init_weights()

        # Define loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Set device
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _init_weights(self):
        """
        Initializes the weights of the network.
        """
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, state):
        """
        Forward pass through the network.

        Parameters:
        - state (torch.Tensor): Input state

        Returns:
        - torch.Tensor: Q-values for each action
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

    def predict(self, state):
        """
        Predicts Q-values for given states.

        Parameters:
        - state (np.ndarray): State input

        Returns:
        - np.ndarray: Predicted Q-values
        """
        self.eval()  # Set network to evaluation mode
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(self.device)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
            q_values = self.forward(state_tensor)
        self.train()  # Set back to training mode
        return q_values.cpu().numpy()

    def train_on_batch(self, state, target):
        """
        Trains the network on a single batch of states and targets.

        Parameters:
        - state (np.ndarray): Batch of states
        - target (np.ndarray): Batch of target Q-values
        """
        self.train()  # Ensure network is in training mode

        # Convert numpy arrays to torch tensors
        state_tensor = torch.from_numpy(state).float().to(self.device)
        target_tensor = torch.from_numpy(target).float().to(self.device)

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.forward(state_tensor)

        # Compute loss
        loss = self.criterion(outputs, target_tensor)

        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()

    def load_weights(self, filepath):
        """
        Loads the network weights from a file.

        Parameters:
        - filepath (str): Path to the saved weights file
        """
        self.load_state_dict(torch.load(filepath, map_location=self.device))
        self.eval()  # Set to evaluation mode after loading
        print(f"Weights loaded from {filepath}")

    def save_weights(self, filepath):
        """
        Saves the network weights to a file.

        Parameters:
        - filepath (str): Path to save the weights file
        """
        torch.save(self.state_dict(), filepath)
        print(f"Weights saved to {filepath}")
