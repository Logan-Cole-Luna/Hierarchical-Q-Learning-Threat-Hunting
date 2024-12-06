"""
Reward calculation module for the reinforcement learning environment.

This module provides functionality to calculate rewards based on actions and labels.

Functions:
    calculate_reward(action, label): Calculates the reward for an action given the true label.
        Returns 1 for correct actions and -1 for incorrect actions.
"""

# utils/reward_calculator.py

import numpy as np
from collections import deque

class RewardCalculator:
    def __init__(self, history_length=10):
        self.correct_streak = 0
        self.history = deque(maxlen=history_length)
        
        # Severity weights for different attack types
        self.severity_weights = {
            0: 1.0,    # Benign
            1: 2.0,    # Botnet
            2: 1.5,    # Brute-force
            3: 2.0,    # DDoS
            4: 1.8,    # DoS
            5: 2.0,    # Infiltration
            6: 1.5     # Web Attack
        }

    def calculate_reward(self, action, true_label, confidence=None):
        """
        Calculate reward based on prediction accuracy, attack severity, and confidence.
        
        Args:
            action (int): Predicted class
            true_label (int): Actual class
            confidence (float, optional): Model's confidence in prediction (0-1)
        
        Returns:
            float: Calculated reward
        """
        # Base reward
        is_correct = (action == true_label)
        base_reward = 1.0 if is_correct else -1.0
        
        # Update streak
        if is_correct:
            self.correct_streak += 1
        else:
            self.correct_streak = 0
            
        # Calculate components
        severity_multiplier = self.severity_weights.get(true_label, 1.0)
        confidence_multiplier = self._calculate_confidence_multiplier(confidence)
        streak_bonus = self._calculate_streak_bonus()
        
        # Combine all factors
        final_reward = (base_reward * 
                       severity_multiplier * 
                       confidence_multiplier + 
                       streak_bonus)
        
        # Update history
        self.history.append(is_correct)
        
        return final_reward
    
    def _calculate_confidence_multiplier(self, confidence):
        """Apply confidence-based scaling if available"""
        if confidence is None:
            return 1.0
        # Penalize low confidence correct predictions and 
        # high confidence incorrect predictions
        return 0.5 + confidence/2

    def _calculate_streak_bonus(self):
        """Calculate bonus for consecutive correct predictions"""
        if self.correct_streak == 0:
            return 0
        return min(0.5 * np.log(self.correct_streak), 2.0)

    def get_performance_stats(self):
        """Return recent performance metrics"""
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)
