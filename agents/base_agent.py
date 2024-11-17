# agents/base_agent.py

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from collections import deque
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Agent:
    def __init__(
        self,
        state_size,
        action_size,
        hidden_layers,
        learning_rate,
        gamma,
        epsilon_start,
        epsilon_end,
        epsilon_decay,
        batch_size,
        memory_size,
        device
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.device = device

        # Initialize the Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, hidden_layers).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_layers).to(device)
        self.optimizer = Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        # Initialize time step for updating target network
        self.t_step = 0
        self.target_update_freq = 1000  # Adjust as needed

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # Shape: [1,63]
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)  # Shape: [1, action_size]
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if np.random.rand() > self.epsilon:
            action = torch.argmax(action_values).item()
        else:
            action = random.choice(np.arange(self.action_size))
        
        logger.debug(f"Action selected: {action} (Epsilon: {self.epsilon})")
        return action

    def learn(self, state, action, reward, next_state, done):
        """Update value parameters using given experience tuple."""
        # Convert to tensors
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # Shape: [1,63]
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)  # Shape: [1,63]
        action = torch.tensor([[action]], dtype=torch.long).to(self.device)  # Shape: [1,1]
        reward = torch.tensor([[reward]], dtype=torch.float).to(self.device)  # Shape: [1,1]
        done = torch.tensor([[done]], dtype=torch.float).to(self.device)  # Shape: [1,1]

        # Log shapes
        logger.debug(f"Learn - state shape: {state.shape}, action shape: {action.shape}, reward shape: {reward.shape}, next_state shape: {next_state.shape}, done shape: {done.shape}")

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)  # Shape: [1,1]
        logger.debug(f"Q_targets_next shape: {Q_targets_next.shape}")

        # Compute Q targets for current states
        Q_targets = reward + (self.gamma * Q_targets_next * (1 - done))  # Shape: [1,1]
        logger.debug(f"Q_targets shape: {Q_targets.shape}")

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(state).gather(1, action)  # Shape: [1,1]
        logger.debug(f"Q_expected shape: {Q_expected.shape}")

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)  # Shape: [1,1]
        logger.debug(f"Loss: {loss.item()}")

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.t_step = (self.t_step + 1) % self.target_update_freq
        if self.t_step == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
            logger.debug("Target network updated.")

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            logger.debug(f"Epsilon decayed to: {self.epsilon}")

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
