"""
Custom OpenAI Gym environment for network intrusion detection.

This environment simulates a network intrusion detection system where an agent must
classify network traffic into various attack types or benign traffic.

Features:
- Batch processing of states
- Customizable episode length
- Support for both binary and multi-class classification
- Built-in reward calculation

Classes:
    NetworkClassificationEnv: Custom Gym environment for network traffic classification.
"""

# utils/intrusion_detection_env.py

import gym
from gym import spaces
import numpy as np
import pandas as pd
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NetworkClassificationEnv(gym.Env):
    def __init__(self, data_df, label_dict, sequence_length=10, batch_size=64, max_steps=1000):
        super(NetworkClassificationEnv, self).__init__()
        self.data_df = data_df.reset_index(drop=True)
        self.label_dict = label_dict
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.current_step = 0
        self.num_samples = len(self.data_df)
        self.labels = None  # Initialize labels
        self.current_state_index = 0  # Initialize current state index
        self.current_sequence = None
        self.sequences = []
        self._create_sequences(data_df)
        
        # Define action and observation space
        self.action_space = spaces.Discrete(len(label_dict))  # Actions correspond to different attack types
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.data_df.shape[1] - 1,), dtype=np.float32
        )
        
        self.reset()

    def _create_sequences(self, data_df):
        """Create sequences from the dataframe."""
        for i in range(0, len(data_df) - self.sequence_length + 1):
            sequence = data_df.iloc[i:i + self.sequence_length]
            self.sequences.append(sequence)

    def reset(self):
        self.current_step = 0
        self.counter = 0  # Counts the number of incorrect actions
        self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)  # Shuffle data
        self.done = False
        self.steps_taken = 0
        self.current_state_index = 0  # Reset state index
        self.current_sequence_idx = 0
        states, labels = self._get_batch()
        self.labels = labels  # Assign labels
        return states, labels

    def _get_batch(self):
        """Get a batch of sequences."""
        end_idx = min(self.current_sequence_idx + self.batch_size, len(self.sequences))
        batch_sequences = self.sequences[self.current_sequence_idx:end_idx]
        
        states = []
        labels = []
        for sequence in batch_sequences:
            # Get features excluding Label and Timestamp
            feature_sequence = sequence.drop(['Label', 'Timestamp', 'Threat'], axis=1).values
            # Use the label of the last timestep
            label = sequence.iloc[-1]['Label']
            states.append(feature_sequence)
            labels.append(label)
        
        self.current_sequence_idx = end_idx
        return np.array(states), np.array(labels)

    def _compute_rewards(self, actions, labels):
        """
        Compute rewards based on the actions and true labels.

        Parameters:
        - actions (list or np.ndarray): Predicted actions
        - labels (np.ndarray): True labels

        Returns:
        - rewards (np.ndarray): Array of rewards
        """
        actions = np.array(actions)
        labels = np.array(labels)
        if actions.shape != labels.shape:
            raise ValueError(f"Shape mismatch: actions shape {actions.shape}, labels shape {labels.shape}")
        rewards = np.where(actions == labels, 1, -1)
        self.counter += np.sum(actions != labels)
        return rewards

    def step(self, actions):
        """
        Execute actions and return the next states, rewards, done flag, and next labels.
        """
        if isinstance(actions, (int, np.integer)):
            actions = [actions]  # Convert single action to list
        actions = np.array(actions)

        # Ensure we have labels for the current step
        if self.labels is None or len(self.labels) == 0:
            logger.warning("No labels available for current step")
            return None, np.array([0]), True, None

        # Get current batch labels
        current_labels = self.labels[:len(actions)]

        # Compute rewards
        rewards = self._compute_rewards(actions, current_labels)

        # Get next batch
        next_states, next_labels = self._get_batch()

        # Update steps and check termination
        self.steps_taken += 1
        done = (self.counter >= self.max_steps or 
                self.current_step >= self.num_samples or 
                self.steps_taken >= self.max_steps)

        # Store labels for next step
        self.labels = next_labels

        # Update current_state_index
        self.current_state_index += len(actions)
        if self.current_state_index >= self.num_samples:
            self.current_state_index = self.num_samples - 1
            done = True

        return next_states, rewards, done, next_labels

    def get_initial_state(self):
        """
        Returns the initial state for evaluation.

        Returns:
        - state (np.ndarray): The initial state.
        """
        state = self.data_df.iloc[0].drop('Label').values.astype(np.float32)
        self.current_state_index = 0
        return state

    def get_true_label(self, state):
        """
        Retrieves the true label for a given state.

        Parameters:
        - state (np.ndarray): The current state.

        Returns:
        - label (int): The true label associated with the state.
        """
        if self.current_state_index < len(self.data_df):
            true_label = self.data_df.iloc[self.current_state_index]['Label']
            # Map string label to numeric using label_dict
            if isinstance(true_label, str):
                true_label = self.label_dict[true_label]
            return true_label
        else:
            logger.warning("Reached end of dataset during evaluation")
            return None
