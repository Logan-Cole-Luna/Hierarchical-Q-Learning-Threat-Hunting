# RewardCalculator.py

class RewardCalculator:
    def __init__(self, category_to_id, anomaly_to_id, benign_id, 
                 high_reward=1.0, low_reward=-0.5, benign_reward=0.1, epsilon=1e-6):
        self.category_to_id = category_to_id
        self.anomaly_to_id = anomaly_to_id
        self.benign_id = benign_id
        self.high_reward = high_reward
        self.low_reward = low_reward
        self.benign_reward = benign_reward
        self.epsilon = epsilon
    
    def calculate_rewards(self, high_pred, high_true, high_confidence, low_pred, low_true, low_confidence):
        high_reward = 0.0
        low_reward = 0.0
        
        # High-Level Reward Calculation
        if high_pred == high_true:
            if high_true == self.benign_id:
                # Correctly predicted benign
                high_reward = self.benign_reward * high_confidence
                low_reward = 0.0  # No action needed for benign
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
            low_reward = 0.0  # Do not penalize Low-Level network
        
        return high_reward, low_reward
