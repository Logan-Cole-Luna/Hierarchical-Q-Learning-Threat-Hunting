# utils/intrusion_detection_env.py

import gym
from gym import spaces
import numpy as np
import pandas as pd
import random

class NetworkClassificationEnv(gym.Env):
    def __init__(self, data_df, label_dict, batch_size=64, max_steps=1000):
        super(NetworkClassificationEnv, self).__init__()
        self.data_df = data_df.reset_index(drop=True)
        self.label_dict = label_dict
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.current_step = 0
        self.num_samples = len(self.data_df)
        
        # Define action and observation space
        self.action_space = spaces.Discrete(len(label_dict))  # Actions correspond to different attack types
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.data_df.shape[1] - 1,), dtype=np.float32
        )
        
        self.reset()

    def reset(self):
        self.current_step = 0
        self.counter = 0  # Counts the number of incorrect actions
        self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)  # Shuffle data
        self.done = False
        self.steps_taken = 0
        return self._get_batch()

    def _get_batch(self):
        end_index = min(self.current_step + self.batch_size, self.num_samples)
        batch_df = self.data_df.iloc[self.current_step:end_index]
        states = batch_df.drop(['Label'], axis=1).values.astype(np.float32)
        labels = batch_df['Label'].map(self.label_dict).values
        self.current_step = end_index
        return states, labels

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

        Parameters:
        - actions (list or np.ndarray): Predicted actions for the current batch

        Returns:
        - next_states (np.ndarray): Next batch of states
        - rewards (np.ndarray): Rewards for the current actions
        - done (bool): Whether the episode is done
        - next_labels (np.ndarray): True labels for the next batch
        """
        rewards = self._compute_rewards(actions, self.labels)
        next_states, next_labels = self._get_batch()
        
        self.steps_taken += 1
        done = False
        if self.counter >= self.max_steps or self.current_step >= self.num_samples or self.steps_taken >= self.max_steps:
            done = True
        
        self.labels = next_labels  # Update labels for the next step
        return next_states, rewards, done, next_labels
