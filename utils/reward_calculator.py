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
                 high_reward=1.0, low_reward=-0.5, benign_reward=0.1, epsilon=1e-6):
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
            Reward for correct high-level predictions.
        low_reward : float, optional
            Penalty for incorrect predictions.
        benign_reward : float, optional
            Reward for correct benign predictions.
        epsilon : float, optional
            Small constant to prevent division by zero in confidence calculations.
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
        Calculates rewards for both high-level and low-level predictions based on accuracy and confidence.

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
            (high_reward, low_reward) based on accuracy and confidence.
        """
        high_reward = 0.0
        low_reward = 0.0
        
        # High-Level Reward Calculation
        if high_pred == high_true:
            if high_true == self.benign_id:
                # Correctly predicted benign case
                high_reward = self.benign_reward * high_confidence
                low_reward = 0.0  # No low-level action needed for benign case
            else:
                # Correctly predicted attack type
                high_reward = self.high_reward * high_confidence
                # Low-Level Reward Calculation
                if low_pred == low_true:
                    low_reward = self.high_reward * low_confidence
                else:
                    low_reward = self.low_reward * low_confidence
        else:
            # High-Level Incorrect Prediction
            high_reward = self.low_reward * (1 - high_confidence)
            low_reward = 0.0  # Do not penalize Low-Level network for high-level errors
        
        return high_reward, low_reward
