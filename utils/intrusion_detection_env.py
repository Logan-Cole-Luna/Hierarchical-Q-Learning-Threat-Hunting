"""
Defines the IntrusionDetectionEnv class for a custom reinforcement learning environment designed
for intrusion detection. The environment includes high-level and low-level feature spaces, along with
category and anomaly mappings for different actions the agent can take.

Classes:
    - IntrusionDetectionEnv: Custom RL environment for handling intrusion detection scenarios.
"""

import numpy as np
import json
from utils.reward_calculator import RewardCalculator

class IntrusionDetectionEnv:
    def __init__(self, X_high, X_low, y_high, y_low, high_agent_action_space, low_agent_action_space, mappings_dir, history_length=1):
        """
        Initializes the environment with feature matrices, labels, action spaces, and mappings.

        Parameters:
        -----------
        X_high : np.ndarray
            Feature matrix for high-level actions.
        X_low : np.ndarray
            Feature matrix for low-level actions.
        y_high : np.ndarray
            High-level labels for each sample.
        y_low : np.ndarray
            Low-level labels for each sample.
        high_agent_action_space : int
            Total number of high-level actions available to the agent.
        low_agent_action_space : int
            Total number of low-level actions available to the agent.
        mappings_dir : str
            Directory path for JSON files mapping categories and anomalies to IDs.
        history_length : int, optional
            Number of previous states to include in the current state (default is 1).
        """
        self.X_high = X_high
        self.X_low = X_low
        self.y_high = y_high
        self.y_low = y_low
        self.num_samples = len(y_high)
        self.current_step = 0
        self.done = False

        self.high_agent_action_space = high_agent_action_space
        self.low_agent_action_space = low_agent_action_space

        # Load category and anomaly mappings
        with open(f"{mappings_dir}/category_to_id.json", 'r') as f:
            self.category_to_id = json.load(f)
        with open(f"{mappings_dir}/anomaly_to_id.json", 'r') as f:
            self.anomaly_to_id = json.load(f)

        # Extract benign_id from category_to_id
        if 'BENIGN' in self.category_to_id:
            self.benign_id = self.category_to_id['BENIGN']
        else:
            raise ValueError("BENIGN category not found in category_to_id mapping.")

        # Initialize RewardCalculator with benign_id
        self.reward_calculator = RewardCalculator(
            self.category_to_id, 
            self.anomaly_to_id, 
            self.benign_id,
            high_reward=1.0, 
            low_reward=-1.0, 
            benign_reward=1.0, 
            epsilon=1e-6
        )

    def reset(self):
        """
        Resets the environment to the beginning of a new episode.

        Returns:
        --------
        tuple
            Initial state as a tuple of high-level and low-level features and labels.
        """
        self.current_step = 0
        self.done = False

        # Shuffle data to simulate dynamic environment
        permutation = np.random.permutation(self.num_samples)
        self.X_high = self.X_high[permutation]
        self.X_low = self.X_low[permutation]
        self.y_high = self.y_high[permutation]
        self.y_low = self.y_low[permutation]

        return self._get_state()

    def _get_state(self):
        """
        Retrieves the current state of the environment.

        Returns:
        --------
        tuple or None
            Tuple (state_high, state_low, high_label, low_label) if within bounds, or None if episode is done.
        """
        if self.current_step < self.num_samples:
            state_high = self.X_high[self.current_step]
            state_low = self.X_low[self.current_step]
            high_label = self.y_high[self.current_step]
            low_label = self.y_low[self.current_step]
            return state_high, state_low, high_label, low_label
        else:
            self.done = True
            return None

    def step(self, action_high, action_low, high_confidence=1.0, low_confidence=1.0):
        """
        Executes one step within the environment, given high and low-level actions.

        Parameters:
        -----------
        action_high : int
            The high-level action selected by the agent (corresponds to a category).
        action_low : int
            The low-level action selected by the agent (corresponds to an anomaly).
        high_confidence : float, optional
            Confidence level of the high-level action (default is 1.0).
        low_confidence : float, optional
            Confidence level of the low-level action (default is 1.0).

        Returns:
        --------
        tuple
            - Current state as (state_high, state_low, high_label, low_label).
            - Reward tuple (reward_high, reward_low) for high and low-level actions.
            - Boolean indicating if the episode has finished.
        """
        if self.done:
            raise Exception("Episode has finished. Call reset() to start a new episode.")

        state = self._get_state()
        if state is None:
            return None, (0.0, 0.0), self.done

        state_high, state_low, high_label, low_label = state

        # Calculate rewards using the RewardCalculator
        reward_high, reward_low = self.reward_calculator.calculate_rewards(
            high_pred=action_high,
            high_true=high_label,
            high_confidence=high_confidence,
            low_pred=action_low,
            low_true=low_label,
            low_confidence=low_confidence
        )

        # Update step counter and check if the episode is complete
        self.current_step += 1
        if self.current_step >= self.num_samples:
            self.done = True

        # Get the next state
        next_state = self._get_state()

        return (next_state if not self.done else None), (reward_high, reward_low), self.done
