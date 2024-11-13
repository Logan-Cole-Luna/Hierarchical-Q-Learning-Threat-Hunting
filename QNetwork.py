# QNetwork.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, learning_rate=0.001):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()  # Using MSELoss for Q-learning
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
    
    def predict(self, state):
        self.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.forward(state)
            probabilities = F.softmax(q_values, dim=1)
        return q_values.cpu().numpy(), probabilities.cpu().numpy()
    
    def train_on_batch(self, states, targets):
        self.train()
        states = torch.FloatTensor(states).to(self.device)
        targets = torch.FloatTensor(targets).to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.forward(states)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_weights(self, filepath):
        torch.save(self.state_dict(), filepath)
    
    def load_weights(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=self.device))
        self.eval()
