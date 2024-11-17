# utils/intrusion_detection_env.py

import gym
from gym import spaces
import numpy as np
import pandas as pd

class NetworkClassificationEnv(gym.Env):
    def __init__(self, data_df, label_dict, batch_size=64):
        super(NetworkClassificationEnv, self).__init__()
        self.data_df = data_df.reset_index(drop=True)
        self.label_dict = label_dict
        self.batch_size = batch_size
        self.current_index = 0
        self.num_samples = len(self.data_df)
        
        # Define action and observation space
        self.action_space = spaces.Discrete(len(label_dict))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.data_df.shape[1]-2,), dtype=np.float32
        )
        self.done = False

    def reset(self):
        self.current_index = 0
        self.done = False
        states, labels = self._get_batch()
        return states, labels

    def step(self, actions):
        # Ensure actions and labels have the same length
        rewards = self._compute_rewards(actions, self.data_df.iloc[self.current_index:self.current_index + self.batch_size]['Label'].map(self.label_dict).values)
        self.current_index += self.batch_size
        if self.current_index >= self.num_samples:
            self.done = True
        next_states, _ = self._get_batch()
        return next_states, rewards, self.done, {}

    def _get_batch(self):
        end_index = min(self.current_index + self.batch_size, self.num_samples)
        batch_df = self.data_df.iloc[self.current_index:end_index]
        states = batch_df.drop(['Label', 'Threat'], axis=1).values.astype(np.float32)
        labels = batch_df['Label'].map(self.label_dict).values
        return states, labels

    def _compute_rewards(self, actions, labels):
        rewards = np.where(actions == labels, 1, -1)
        return rewards
