# agents/base_agent.py

import torch
import torch.optim as optim  # Ensure this import is present
import torch.nn.functional as F
import numpy as np
from utils.q_network import QNetwork, save_model, load_model
from collections import deque
import random
import json

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class BaseAgent:
    def __init__(self, state_size, action_size, hidden_layers, lr, gamma, epsilon, epsilon_min, epsilon_decay, memory_capacity, batch_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device

        self.memory = ReplayMemory(memory_capacity)
        self.model = QNetwork(state_size, action_size, hidden_layers).to(self.device)
        self.target_model = QNetwork(state_size, action_size, hidden_layers).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # Ensure 'optim' is defined

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values, dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0.0  # No training if memory is insufficient

        minibatch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert lists of arrays to single NumPy arrays
        states = np.array(states)
        next_states = np.array(next_states)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q values
        current_q = self.model(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        # Compute loss
        loss = F.mse_loss(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, path):
        save_model(self.model, path)

    def load(self, path):
        load_model(self.model, path)
        self.update_target_model()
