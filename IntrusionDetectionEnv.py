# IntrusionDetectionEnv.py

import numpy as np
import json
from RewardCalculator import RewardCalculator

class IntrusionDetectionEnv:
    def __init__(self, X_high, X_low, y, high_agent_action_space, low_agent_action_space, mappings_dir):
        self.X_high = X_high
        self.X_low = X_low
        self.y = y
        self.num_samples = len(y)
        self.current_step = 0
        self.high_agent_action_space = high_agent_action_space  # Number of categories
        self.low_agent_action_space = low_agent_action_space    # Number of anomalies
        self.done = False
        
        # Load mappings
        with open(f"{mappings_dir}/category_to_id.json", 'r') as f:
            self.category_to_id = json.load(f)
        with open(f"{mappings_dir}/anomaly_to_id.json", 'r') as f:
            self.anomaly_to_id = json.load(f)
        
        # Create inverse mappings for reference (optional)
        self.id_to_category = {v: k for k, v in self.category_to_id.items()}
        self.id_to_anomaly = {v: k for k, v in self.anomaly_to_id.items()}
        
        # Initialize RewardCalculator
        self.reward_calculator = RewardCalculator(self.category_to_id, self.anomaly_to_id)
    
    def reset(self):
        self.current_step = 0
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        if self.current_step < self.num_samples:
            state_high = self.X_high[self.current_step]
            state_low = self.X_low[self.current_step]
            label = self.y[self.current_step]
            return state_high, state_low, label
        else:
            self.done = True
            return None
    
    def step(self, action_high, action_low):
        if self.done:
            raise Exception("Episode has finished. Call reset() to start a new episode.")
        
        state_high, state_low, label = self._get_state()
        
        # Calculate rewards using RewardCalculator
        reward_high = self.reward_calculator.calculate_high_reward(action_high, label)
        reward_low = self.reward_calculator.calculate_low_reward(action_low, label)
        
        # Update step
        self.current_step += 1
        if self.current_step >= self.num_samples:
            self.done = True
        
        next_state = self._get_state()
        return (state_high, state_low, label), (reward_high, reward_low), self.done
