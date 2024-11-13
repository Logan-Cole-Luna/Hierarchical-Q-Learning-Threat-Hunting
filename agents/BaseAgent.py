# agents/BaseAgent.py

import numpy as np
import random
from collections import deque
from QNetwork import QNetwork

class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.memory = deque(maxlen=capacity)
    
    def add(self, experience):
        self.memory.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class BaseAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma    # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = ReplayBuffer(capacity=2000)
        self.q_network = QNetwork(state_size, action_size, learning_rate)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = self.memory.sample(self.batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.q_network.predict(next_state)[0])
            target_f = self.q_network.predict(state)
            target_f[0][action] = target
            states.append(state[0])
            targets.append(target_f[0])
        states = np.array(states)
        targets = np.array(targets)
        self.q_network.train_on_batch(states, targets)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.q_network.model.load_weights(name)
    
    def save(self, name):
        self.q_network.model.save_weights(name)
