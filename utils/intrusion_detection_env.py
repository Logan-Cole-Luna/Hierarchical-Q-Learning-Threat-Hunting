# utils/intrusion_detection_env.py

import gym
from gym import spaces
import numpy as np
import pandas as pd
import json
import torch

class IntrusionDetectionEnv(gym.Env):
    def __init__(self, X_high, X_low, y_high, y_low, category_to_id, anomaly_to_id, history_length=1):
        super(IntrusionDetectionEnv, self).__init__()
        self.X_high = X_high
        self.X_low = X_low
        self.y_high = y_high
        self.y_low = y_low
        self.category_to_id = category_to_id
        self.anomaly_to_id = anomaly_to_id
        self.history_length = history_length

        self.current_index = 0
        self.done = False

        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.category_to_id))  # High-level actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.X_high.shape[1] + self.X_low.shape[1],), dtype=np.float32)

    def reset(self):
        self.current_index = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        if self.current_index >= len(self.X_high):
            self.done = True
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        high_state = self.X_high[self.current_index]
        low_state = self.X_low[self.current_index]
        observation = np.concatenate([high_state, low_state]).astype(np.float32)
        return observation

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, self.done, {}

        # Get true label for high-level action
        true_action = self.y_high[self.current_index]

        # Calculate reward
        reward = 1.0 if action == true_action else -1.0

        # Move to next state
        self.current_index += 1
        if self.current_index >= len(self.X_high):
            self.done = True

        observation = self._get_observation()
        return observation, reward, self.done, {}

    def render(self, mode='human'):
        pass  # Implement visualization if needed
