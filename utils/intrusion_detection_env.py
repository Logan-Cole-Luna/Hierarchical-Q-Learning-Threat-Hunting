"""
intrusion_detection_env.py

Defines the IntrusionDetectionEnv class for a custom reinforcement learning environment designed
for intrusion detection. The environment includes high-level and low-level feature spaces, along with
category and anomaly mappings for different actions the agent can take.

Classes:
    - IntrusionDetectionEnv: Custom RL environment for handling intrusion detection scenarios.
"""

import numpy as np
import json
from utils.reward_calculator import RewardCalculator
from collections import deque

class IntrusionDetectionEnv:
    def __init__(self, X_high, X_low, y_high, y_low, high_agent_action_space, low_agent_action_space, mappings_dir, history_length=4):
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
        history_length : int, optional
            Number of previous states to include in the current state (default is 4).
        """
        self.X_high = X_high
        self.X_low = X_low
        self.y_high = y_high
        self.y_low = y_low
        self.num_samples = len(y_high)
        self.current_step = 0
        self.done = False

        # Internal state to simulate an alert level for sequential decision-making
        self.internal_state = {'alert_level': 0}

        # Action space sizes for high and low-level agents
        self.high_agent_action_space = high_agent_action_space
        self.low_agent_action_space = low_agent_action_space

        # Load category and anomaly mappings
        with open(f"{mappings_dir}/category_to_id.json", 'r') as f:
            self.category_to_id = json.load(f)
        with open(f"{mappings_dir}/anomaly_to_id.json", 'r') as f:
            self.anomaly_to_id = json.load(f)

        self.id_to_category = {v: k for k, v in self.category_to_id.items()}
        self.id_to_anomaly = {v: k for k, v in self.anomaly_to_id.items()}

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
            low_reward=-0.5, 
            benign_reward=0.1, 
            epsilon=1e-6
        )

        # Initialize history buffers for high and low-level states
        self.history_length = history_length
        self.state_history_high = deque(maxlen=history_length)
        self.state_history_low = deque(maxlen=history_length)

        # Initialize history with zeros
        zero_high = np.zeros(self.X_high.shape[1])
        zero_low = np.zeros(self.X_low.shape[1])
        for _ in range(history_length):
            self.state_history_high.append(zero_high)
            self.state_history_low.append(zero_low)

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

        # Shuffle data to simulate dynamic environment
        permutation = np.random.permutation(self.num_samples)
        self.X_high = self.X_high[permutation]
        self.X_low = self.X_low[permutation]
        self.y_high = self.y_high[permutation]
        self.y_low = self.y_low[permutation]

        # Reset state histories
        self.state_history_high = deque(maxlen=self.history_length)
        self.state_history_low = deque(maxlen=self.history_length)
        zero_high = np.zeros(self.X_high.shape[1])
        zero_low = np.zeros(self.X_low.shape[1])
        for _ in range(self.history_length):
            self.state_history_high.append(zero_high)
            self.state_history_low.append(zero_low)

        return self._get_state()

    def _get_state(self):
        """
        Retrieves the current state of the environment.

        Returns:
        --------
        tuple or None
            Tuple (combined_state_high, combined_state_low, high_label, low_label) if within bounds, or None if episode is done.
        """
        if self.current_step < self.num_samples:
            state_high = self.X_high[self.current_step]
            state_low = self.X_low[self.current_step]
            high_label = self.y_high[self.current_step]
            low_label = self.y_low[self.current_step]

            # Update state history
            self.state_history_high.append(state_high)
            self.state_history_low.append(state_low)

            # Concatenate history with internal state
            combined_state_high = np.concatenate(
                list(self.state_history_high) + [np.array([self.internal_state['alert_level']])]
            )
            combined_state_low = np.concatenate(
                list(self.state_history_low) + [np.array([self.internal_state['alert_level']])]
            )

            return combined_state_high, combined_state_low, high_label, low_label
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
            - Current state as (combined_state_high, combined_state_low, high_label, low_label).
            - Reward tuple (reward_high, reward_low) for high and low-level actions.
            - Boolean indicating if the episode has finished.
        """
        if self.done:
            raise Exception("Episode has finished. Call reset() to start a new episode.")

        state = self._get_state()
        if state is None:
            return None, (0.0, 0.0), self.done

        combined_state_high, combined_state_low, high_label, low_label = state

        # Calculate rewards using the RewardCalculator
        reward_high, reward_low = self.reward_calculator.calculate_rewards(
            high_pred=action_high,
            high_true=high_label,
            high_confidence=high_confidence,
            low_pred=action_low,
            low_true=low_label,
            low_confidence=low_confidence
        )

        # Update internal state based on actions
        if action_high != high_label:
            self.internal_state['alert_level'] += 1  # Increase alert level on misclassification
        else:
            self.internal_state['alert_level'] = max(0, self.internal_state['alert_level'] - 1)  # Decrease alert level on correct classification

        # Clip alert level to a maximum value
        self.internal_state['alert_level'] = min(self.internal_state['alert_level'], 5)

        # Update step counter and check if the episode is complete
        self.current_step += 1
        if self.current_step >= self.num_samples:
            self.done = True

        # Get the next state
        next_state = self._get_state()

        return (combined_state_high, combined_state_low, high_label, low_label), (reward_high, reward_low), self.done
