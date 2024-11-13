"""
LowLevelAgent.py

Defines the `LowLevelAgent` class, a specialized agent inheriting from `BaseAgent`.
The `LowLevelAgent` is used for low-level actions within a hierarchical reinforcement learning framework.

Classes:
    - LowLevelAgent: Inherits from BaseAgent and specializes for low-level action tasks.
"""

from .BaseAgent import BaseAgent

class LowLevelAgent(BaseAgent):
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64):
        """
        Initializes the LowLevelAgent with specified parameters, inheriting 
        from BaseAgent for Q-learning behavior.

        Parameters:
        -----------
        state_size : int
            Dimension of the state space.
        action_size : int
            Number of possible low-level actions.
        learning_rate : float, optional
            Learning rate for the agent's Q-network (default is 0.001).
        gamma : float, optional
            Discount factor for future rewards (default is 0.95).
        epsilon : float, optional
            Initial exploration rate for epsilon-greedy policy (default is 1.0).
        epsilon_min : float, optional
            Minimum exploration rate (default is 0.01).
        epsilon_decay : float, optional
            Factor by which epsilon is reduced after each episode (default is 0.995).
        batch_size : int, optional
            Number of experiences to sample during training (default is 64).
        """
        # Initialize parent class (BaseAgent) with provided parameters
        super().__init__(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size
        )
        # Additional initialization specific to LowLevelAgent can be added here if required
