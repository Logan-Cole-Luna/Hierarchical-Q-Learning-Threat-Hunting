# agents/policies.py

import random
import numpy as np
import torch

class EpsilonGreedy:
    def __init__(self, action_size, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.action_size = action_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
    
    def get_action(self, state, qnetwork, device):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        qnetwork.eval()
        with torch.no_grad():
            action_values = qnetwork(state)
        qnetwork.train()
        
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)
