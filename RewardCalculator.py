"""
RewardCalculator.py

Defines the `RewardCalculator` class for calculating rewards in a reinforcement learning environment
based on high-level and low-level actions. Rewards are determined by comparing the agent's 
predicted actions with the true labels for categories and anomalies.

Classes:
    - RewardCalculator: Provides methods to calculate rewards for high-level and low-level actions.
"""

class RewardCalculator:
    def __init__(self, category_to_id, anomaly_to_id):
        """
        Initializes the RewardCalculator with mappings for categories and anomalies.

        Parameters:
        -----------
        category_to_id : dict
            Mapping of category names to unique IDs.
        anomaly_to_id : dict
            Mapping of anomaly types to unique IDs.
        """
        self.category_to_id = category_to_id
        self.anomaly_to_id = anomaly_to_id
    
    def calculate_high_reward(self, action_high, label):
        """
        Calculates the reward for a high-level action by comparing the predicted category with the true label.

        Parameters:
        -----------
        action_high : int
            The predicted category ID by the high-level agent.
        label : int
            The true category ID.

        Returns:
        --------
        int
            Reward value: +1 if the prediction is correct, -1 otherwise.
        """
        # Reward +1 for correct category prediction, -1 otherwise
        return 1 if action_high == label else -1
    
    def calculate_low_reward(self, action_low, label):
        """
        Calculates the reward for a low-level action by comparing the predicted anomaly with the true label.

        Parameters:
        -----------
        action_low : int
            The predicted anomaly ID by the low-level agent.
        label : int
            The true anomaly ID.

        Returns:
        --------
        int
            Reward value: +1 if the prediction is correct, -1 otherwise.
        """
        true_anomaly = label  # Assuming label corresponds to anomaly ID
        # Reward +1 for correct anomaly prediction, -1 otherwise
        return 1 if action_low == true_anomaly else -1
