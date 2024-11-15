"""
BaseAgent.py

Defines the `ReplayBuffer` and `BaseAgent` classes for a reinforcement learning agent. 
The `ReplayBuffer` handles storage and sampling of experience tuples for training, 
while `BaseAgent` manages action selection, experience replay, and network training.

Classes:
    - ReplayBuffer: Stores agent's experiences for training.
    - BaseAgent: Reinforcement learning agent with experience replay and Q-learning.
"""

import numpy as np
import random
from collections import deque
from utils.q_network import QNetwork

class ReplayBuffer:
    def __init__(self, capacity=2000):
        """
        Initializes the replay buffer to store experiences.

        Parameters:
        -----------
        capacity : int, optional
            Maximum number of experiences to store (default is 2000).
        """
        self.memory = deque(maxlen=capacity)

    def add(self, experience):
        """
        Adds a new experience to the buffer.

        Parameters:
        -----------
        experience : tuple
            A tuple of (state, action, reward, next_state, done).
        """
        self.memory.append(experience)

    def sample(self, batch_size):
        """
        Samples a random batch of experiences from the buffer.

        Parameters:
        -----------
        batch_size : int
            Number of experiences to sample.

        Returns:
        --------
        list
            List of randomly selected experiences.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Returns the current size of the replay buffer.
        """
        return len(self.memory)

class BaseAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64):
        """
        Initializes the base agent with hyperparameters and Q-network.

        Parameters:
        -----------
        state_size : int
            Dimension of the state space.
        action_size : int
            Number of possible actions the agent can take.
        learning_rate : float, optional
            Learning rate for Q-network training (default is 0.001).
        gamma : float, optional
            Discount factor for future rewards (default is 0.95).
        epsilon : float, optional
            Initial exploration rate for epsilon-greedy policy (default is 1.0).
        epsilon_min : float, optional
            Minimum exploration rate (default is 0.01).
        epsilon_decay : float, optional
            Factor by which epsilon is reduced each episode (default is 0.995).
        batch_size : int, optional
            Number of experiences to sample during training (default is 64).
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Initialize replay buffer and Q-networks
        self.memory = ReplayBuffer(capacity=2000)
        self.q_network = QNetwork(state_size, action_size, learning_rate)
        self.target_network = QNetwork(state_size, action_size, learning_rate)
        self.update_target_network()

    def update_target_network(self):
        """
        Updates the target network's weights to match the main Q-network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple in the replay buffer.

        Parameters:
        -----------
        state : array-like
            Current state.
        action : int
            Action taken by the agent.
        reward : float
            Reward received after taking the action.
        next_state : array-like
            Next state after the action.
        done : bool
            Flag indicating whether the episode ended.
        """
        self.memory.add((state, action, reward, next_state, done))

    def act(self, state):
        """
        Chooses an action based on epsilon-greedy policy.

        Parameters:
        -----------
        state : array-like
            Current state to evaluate.

        Returns:
        --------
        int
            Chosen action.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def replay(self):
        """
        Trains the agent using experiences from the replay buffer. 
        Performs a Q-learning update using a minibatch of past experiences.
        """
        if len(self.memory) < self.batch_size:
            return

        minibatch = self.memory.sample(self.batch_size)
        states = []
        targets = []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Use the target network for more stable target Q-value estimates
                target += self.gamma * np.amax(self.target_network.predict(next_state)[0])

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
        """
        Loads the agent's model weights from a specified file.

        Parameters:
        -----------
        name : str
            Path to the file containing the model weights.
        """
        self.q_network.load_weights(name)
        self.target_network.load_weights(name)

    def save(self, name):
        """
        Saves the agent's model weights to a specified file.

        Parameters:
        -----------
        name : str
            Path where the model weights will be saved.
        """
        self.q_network.save_weights(name)
        self.target_network.save_weights(name)
