# agents/base_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
import logging

from agents.replay_memory import ReplayMemory
from utils.q_network import QNetwork

logger = logging.getLogger(__name__)

# Define a namedtuple for storing experiences in replay memory
Experience = namedtuple('Experience',
                        field_names=['state', 'action', 'reward', 'next_state', 'done'])

class Agent:
    def __init__(
        self,
        state_size,
        action_size,
        hidden_layers=[64, 32],
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        memory_size=10000,
        device=torch.device("cpu")
    ):
        """
        Initializes the Agent with given parameters.

        Parameters:
        - state_size (int): Dimension of each state
        - action_size (int): Number of possible actions
        - hidden_layers (list of int): Sizes of hidden layers
        - learning_rate (float): Learning rate for optimizer
        - gamma (float): Discount factor
        - epsilon_start (float): Initial epsilon for epsilon-greedy policy
        - epsilon_end (float): Minimum epsilon
        - epsilon_decay (float): Decay rate for epsilon
        - batch_size (int): Mini-batch size for training
        - memory_size (int): Size of replay memory
        - device (torch.device): Device to run computations on
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device
        
        # Initialize Q-Network and Target Network
        self.qnetwork_local = QNetwork(state_size, action_size, hidden_layers).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_layers).to(self.device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        
        # Replay memory
        self.memory = ReplayMemory(memory_size)
        
        # Initialize time step (for updating target network)
        self.t_step = 0
        self.target_update_freq = 1000  # Update target network every 1000 steps

    def act(self, states):
        """
        Selects actions for a batch of states using an epsilon-greedy policy.

        Parameters:
        - states (np.ndarray): Batch of states with shape [batch_size, state_size]

        Returns:
        - actions (list): List of selected action indices
        """
        states = torch.FloatTensor(states).to(self.device)  # Shape: [batch_size, state_size]
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(states)  # Shape: [batch_size, action_size]
        self.qnetwork_local.train()
        
        batch_size = action_values.size(0)
        
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            # Select the action with the highest Q-value for each state
            actions = torch.argmax(action_values, dim=1).cpu().numpy()
        else:
            # Select random actions for each state
            actions = np.random.randint(0, self.action_size, size=batch_size)
        
        return actions.tolist()  # Ensure it's a list of integers

    def step(self, state, action, reward, next_state, done):
        """
        Saves experience in replay memory and performs learning step.

        Parameters:
        - state (np.ndarray): Current state
        - action (int): Action taken
        - reward (float): Reward received
        - next_state (np.ndarray): Next state
        - done (bool): Whether the episode has ended

        Returns:
        - loss (float): The loss from the learning step (if performed), else None
        """
        # Save experience in replay memory
        self.memory.push(state, action, reward, next_state, done)
        
        # Learn every time step, if enough samples are available in memory
        loss = None
        if len(self.memory) > self.batch_size:
            loss = self.learn()
        
        return loss

    def learn(self):
        """
        Updates value parameters using a batch of experience tuples.

        Returns:
        - loss (float): The computed loss value
        """
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)            # Shape: [batch_size, state_size]
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # Shape: [batch_size, 1]
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device) # Shape: [batch_size, 1]
        next_states = torch.FloatTensor(next_states).to(self.device)      # Shape: [batch_size, state_size]
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)    # Shape: [batch_size, 1]
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)  # Shape: [batch_size, 1]
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)  # Shape: [batch_size, 1]
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.t_step = (self.t_step + 1) % self.target_update_freq
        if self.t_step == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

# Define the Q-Network architecture
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers):
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
        return self.network(state)
