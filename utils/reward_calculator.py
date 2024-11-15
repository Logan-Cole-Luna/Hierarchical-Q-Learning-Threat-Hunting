"""
RewardCalculator.py

Defines the `RewardCalculator` class for calculating rewards in a reinforcement learning environment
based on high-level and low-level actions. The rewards are calculated based on the agent's
confidence in its predictions, accuracy, and a distinction between benign and malicious classifications.

Classes:
    - RewardCalculator: Provides methods to calculate rewards for high-level and low-level actions.
"""

class RewardCalculator:
    def __init__(self, category_to_id, anomaly_to_id, benign_id, 
                 high_reward=1.0, low_reward=-1.0, benign_reward=1.0, epsilon=1e-6):
        """
        Initializes the RewardCalculator with mappings, reward values, and a benign identifier.

        Parameters:
        -----------
        category_to_id : dict
            Mapping of category names to unique IDs.
        anomaly_to_id : dict
            Mapping of anomaly types to unique IDs.
        benign_id : int
            Identifier for benign categories.
        high_reward : float, optional
            Reward for correct high-level predictions (default is +1.0).
        low_reward : float, optional
            Penalty for incorrect predictions (default is -1.0).
        benign_reward : float, optional
            Reward for correct benign predictions (default is +1.0).
        epsilon : float, optional
            Small constant to prevent division by zero in confidence calculations (default is 1e-6).
        """
        self.category_to_id = category_to_id
        self.anomaly_to_id = anomaly_to_id
        self.benign_id = benign_id
        self.high_reward = high_reward
        self.low_reward = low_reward
        self.benign_reward = benign_reward
        self.epsilon = epsilon

    def calculate_rewards(self, high_pred, high_true, high_confidence, low_pred, low_true, low_confidence):
        """
        Calculates rewards for both high-level and low-level predictions based on accuracy.

        Parameters:
        -----------
        high_pred : int
            High-level prediction made by the agent.
        high_true : int
            True high-level label.
        high_confidence : float
            Confidence level of the high-level prediction.
        low_pred : int
            Low-level prediction made by the agent.
        low_true : int
            True low-level label.
        low_confidence : float
            Confidence level of the low-level prediction.

        Returns:
        --------
        tuple
            (reward_high, reward_low) based on accuracy.
        """
        # High-Level Reward
        if high_pred == high_true:
            if high_true == self.benign_id:
                reward_high = self.benign_reward
                reward_low = 0.0  # No low-level action needed for benign case
            else:
                reward_high = self.high_reward
                # Low-Level Reward
                reward_low = self.high_reward if low_pred == low_true else self.low_reward
        else:
            reward_high = self.low_reward
            reward_low = 0.0  # Do not penalize low-level agent for high-level errors

        return reward_high, reward_low
