# utils/reward_calculator.py

import numpy as np

def calculate_reward(action, label):
    return 1 if action == label else -1
