# agents/replay_memory.py

import random
from collections import deque

class ReplayMemory:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity):
        """
        Initialize a ReplayMemory object.

        Parameters:
        - capacity (int): Maximum size of buffer
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.

        Parameters:
        - state (np.ndarray): Current state
        - action (int): Action taken
        - reward (float): Reward received
        - next_state (np.ndarray): Next state
        - done (bool): Whether the episode has ended
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from memory.

        Parameters:
        - batch_size (int): Size of each training batch

        Returns:
        - list of tuples: Sampled experiences
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
