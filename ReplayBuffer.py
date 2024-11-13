"""
replay_buffer.py

Defines the `ReplayBuffer` class for storing and sampling experiences in reinforcement learning.
The buffer is used to store transitions (state, action, reward, next_state, done) and allows 
random sampling to facilitate experience replay for training agents.

Classes:
    - ReplayBuffer: Manages storage and retrieval of experience tuples for reinforcement learning.
"""

import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        """
        Initializes the replay buffer with a specified capacity.

        Parameters:
        -----------
        capacity : int, optional
            Maximum number of experiences to store in the buffer (default is 10,000).
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the replay buffer.

        Parameters:
        -----------
        state : array-like
            The current state observed by the agent.
        action : int
            The action taken by the agent.
        reward : float
            The reward received after taking the action.
        next_state : array-like
            The next state observed after taking the action.
        done : bool
            Flag indicating whether the episode has ended.
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Samples a random batch of experiences from the buffer.

        Parameters:
        -----------
        batch_size : int
            The number of experiences to sample.

        Returns:
        --------
        tuple
            Tuple containing arrays of states, actions, rewards, next_states, and done flags.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        """
        Returns the current size of the replay buffer.

        Returns:
        --------
        int
            Number of experiences currently stored in the buffer.
        """
        return len(self.buffer)
