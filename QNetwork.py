# QNetwork.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, learning_rate=0.001):
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        # Define the network layers with increased capacity
        self.fc1 = nn.Linear(self.state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.action_size)
        
        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Define the loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, state):
        """
        Perform a forward pass through the network.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
    
    def predict(self, state):
        """
        Predict Q-values for given states.
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.forward(state)
        return q_values.cpu().numpy()
    
    def train_on_batch(self, state, target):
        """
        Train the network on a batch of states and target Q-values.
        Returns the computed loss.
        """
        self.train()  # Set the model to training mode
        state = torch.FloatTensor(state).to(self.device)
        target = torch.LongTensor(target).to(self.device)  # Targets should be LongTensor for CrossEntropyLoss
        
        # Forward pass
        q_values = self.forward(state)
        
        # Compute loss
        loss = self.loss_fn(q_values, target)
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()  # Return the loss value
    
    def load_weights(self, filepath):
        """
        Load model weights from a file.
        """
        self.load_state_dict(torch.load(filepath, map_location=self.device))
        self.eval()
        print(f"Model weights loaded from {filepath}")
    
    def save_weights(self, filepath):
        """
        Save model weights to a file.
        """
        torch.save(self.state_dict(), filepath)
        print(f"Model weights saved to {filepath}")
