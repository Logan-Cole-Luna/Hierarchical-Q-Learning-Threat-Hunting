# agents/policies.py

import random
import numpy as np
import torch

class EpsilonGreedyPolicy:
    def __init__(self, agent, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.agent = agent
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.agent.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
        with torch.no_grad():
            q_values = self.agent.model(state)
        return torch.argmax(q_values, dim=1).item()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
