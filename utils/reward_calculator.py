"""
Reward calculation module for the reinforcement learning environment.

This module provides functionality to calculate rewards based on actions and labels.

Functions:
    calculate_reward(action, label): Calculates the reward for an action given the true label.
        Returns 1 for correct actions and -1 for incorrect actions.
"""

# utils/reward_calculator.py

import numpy as np

def calculate_reward(action, label):
    return 1 if action == label else -1
