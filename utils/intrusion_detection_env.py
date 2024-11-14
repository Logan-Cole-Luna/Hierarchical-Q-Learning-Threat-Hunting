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
    def __init__(self, X_high, X_low, y_high, y_low, high_agent_action_space, low_agent_action_space, mappings_dir):
        """
        Initializes the environment with feature matrices, labels, action spaces, and mappings.

        Parameters:
        -----------
        X_high : np.ndarray
            Feature matrix for high-level actions, used to represent broader categories in the data.
        X_low : np.ndarray
            Feature matrix for low-level actions, used for finer-grained anomaly detection.
        y_high : np.ndarray
            High-level labels for each sample, indicating the category of intrusion or normal activity.
        y_low : np.ndarray
            Low-level labels for each sample, indicating specific anomaly types or normal activity.
        high_agent_action_space : int
            Total number of high-level actions available to the agent.
        low_agent_action_space : int
            Total number of low-level actions available to the agent.
        mappings_dir : str
            Directory path for JSON files mapping categories and anomalies to IDs.
        """
        # Store feature matrices and labels
        self.X_high = X_high
        self.X_low = X_low
        self.y_high = y_high
        self.y_low = y_low
        self.num_samples = len(y_high)
        self.current_step = 0
        self.done = False

        # Internal state to simulate an alert level for sequential decision-making
        self.internal_state = {'alert_level': 0}

        self.high_agent_action_space = high_agent_action_space
        self.low_agent_action_space = low_agent_action_space

        # Load category and anomaly mappings
        with open(f"{mappings_dir}/category_to_id.json", 'r') as f:
            self.category_to_id = json.load(f)
        with open(f"{mappings_dir}/anomaly_to_id.json", 'r') as f:
            self.anomaly_to_id = json.load(f)
        
        
        # Create inverse mappings (optional, useful for debugging or displaying actions)

        # Create inverse mappings (optional, useful for debugging or displaying actions)
        self.id_to_category = {v: k for k, v in self.category_to_id.items()}
        self.id_to_anomaly = {v: k for k, v in self.anomaly_to_id.items()}
        
        
        # Initialize the reward calculator with mappings

        # Initialize the reward calculator with mappings
        self.reward_calculator = RewardCalculator(self.category_to_id, self.anomaly_to_id)

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
        self.internal_state['alert_level'] = 0
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

            combined_state_high = np.concatenate((state_high, [self.internal_state['alert_level']]))
            combined_state_low = np.concatenate((state_low, [self.internal_state['alert_level']]))
            return combined_state_high, combined_state_low, high_label, low_label
        else:
            self.done = True
            return None

    def step(self, action_high, action_low):
        """
        Executes one step within the environment, given high and low-level actions.

        Parameters:
        -----------
        action_high : int
            The high-level action selected by the agent (corresponds to a category).
        action_low : int
            The low-level action selected by the agent (corresponds to an anomaly).

        Returns:
        --------
        tuple
            - Current state as (state_high, state_low, high_label, low_label).
            - Reward tuple (reward_high, reward_low) for high and low-level actions.
            - Boolean indicating if the episode has finished.
        """
        if self.done:
            raise Exception("Episode has finished. Call reset() to start a new episode.")
        
        
        # Retrieve current state and labels

        # Retrieve current state and labels
        state_high, state_low, high_label, low_label = self._get_state()

        reward_high = self.reward_calculator.calculate_high_reward(action_high, high_label, self.internal_state)
        reward_low = self.reward_calculator.calculate_low_reward(action_low, low_label, self.internal_state)

        if action_high != high_label:
            self.internal_state['alert_level'] += 1
        else:
            self.internal_state['alert_level'] = max(0, self.internal_state['alert_level'] - 1)

        self.internal_state['alert_level'] = min(self.internal_state['alert_level'], 5)

        self.current_step += 1
        if self.current_step >= self.num_samples:
            self.done = True
        
        
        # Get the next state

        # Get the next state
        next_state = self._get_state()
        return (state_high, state_low, high_label, low_label), (reward_high, reward_low), self.done
