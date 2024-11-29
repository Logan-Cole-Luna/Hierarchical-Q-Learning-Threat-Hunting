"""
Base reinforcement learning agent implementation.

This module implements a DQN (Deep Q-Network) agent with the following features:
- Experience replay
- Target network for stable learning
- Epsilon-greedy exploration
- Configurable network architecture
- Batch prediction support

The agent can handle both binary and multi-class classification tasks through
its Q-network architecture.

Classes:
    Agent: Implementation of DQN agent with experience replay and target networks.
"""

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

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers, sequence_length):
        super(QNetwork, self).__init__()
        self.lstm = nn.LSTM(
            input_size=state_size,
            hidden_size=hidden_layers[0],
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_layers[i], hidden_layers[i+1])
            for i in range(len(hidden_layers)-1)
        ])
        
        self.output = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, state_size]
        lstm_out, _ = self.lstm(x)
        # Use only the last timestep's output
        x = lstm_out[:, -1, :]
        
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        return self.output(x)

class Agent:
    def __init__(
        self,
        state_size,
        action_size,
        sequence_length=10,
        hidden_layers=[128, 64],
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
        self.action_size = action_size  # Ensure action_size matches the number of classes
        self.sequence_length = sequence_length
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device
        
        # Initialize Q-Network and Target Network
        self.qnetwork_local = QNetwork(state_size, action_size, hidden_layers, sequence_length).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_layers, sequence_length).to(self.device)
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
        - states (list or np.ndarray): Batch of states with shape [batch_size, sequence_length, state_size]

        Returns:
        - actions (list): List of selected action indices
        """
        states = torch.FloatTensor(states).to(self.device)  # Shape: [batch_size, sequence_length, state_size]
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(states)  # Shape: [batch_size, action_size]
        self.qnetwork_local.train()

        batch_size = action_values.shape[0]

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            # Select the action with the highest Q-value for each state
            actions = np.argmax(action_values.cpu().numpy(), axis=1)
        else:
            # Select random actions for each state
            actions = np.random.randint(0, self.action_size, size=batch_size)

        return actions.tolist()

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
        
        # Convert to tensors and move to device
        states = np.array(states)  # Convert list of arrays to a single NumPy array
        states = torch.FloatTensor(states).to(self.device)            # Shape: [batch_size, sequence_length, state_size]
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # Shape: [batch_size, 1]
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device) # Shape: [batch_size, 1]
        next_states = np.array(next_states)  # Convert list of arrays to a single NumPy array
        next_states = torch.FloatTensor(next_states).to(self.device)      # Shape: [batch_size, sequence_length, state_size]
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)    # Shape: [batch_size, 1]
        
        # Get max predicted Q values (for next states) from target model
        with torch.no_grad():
            Q_targets_next = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)  # Shape: [batch_size, 1]
        
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

    def predict_batch(self, states):
        """
        Predicts actions and returns both predictions and Q-values for a batch of states.
        
        Parameters:
        - states (list or np.ndarray): Batch of input states

        Returns:
        - preds (list): List of predicted action indices.
        - q_values (np.ndarray): Q-values for each action, shape [batch_size, num_actions].
        """
        self.qnetwork_local.eval()
        with torch.no_grad():
            states = np.array(states)  # Convert list of arrays to a single NumPy array
            states = torch.FloatTensor(states).to(self.device)  # Shape: [batch_size, sequence_length, state_size]
            q_values = self.qnetwork_local(states)                        # Shape: [batch_size, num_actions]
            q_values = q_values.cpu().numpy()
        self.qnetwork_local.train()
        preds = np.argmax(q_values, axis=1)
        return preds.tolist(), q_values   # Ensure q_values is a 2D numpy array

    def get_action_probabilities(self, state):
        """
        Get softmax probabilities for each action given a state.
        
        Parameters:
        - state (torch.Tensor): The current state tensor
        
        Returns:
        - action_probs (torch.Tensor): Probability distribution over actions
        """
        with torch.no_grad():
            # Get action values from the network
            action_values = self.qnetwork_local(state)
            
            # Convert to probabilities using softmax
            action_probs = torch.softmax(action_values, dim=1)
            
        return action_probs

