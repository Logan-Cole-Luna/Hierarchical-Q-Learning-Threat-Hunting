# RewardCalculator.py

class RewardCalculator:
    def __init__(self, category_to_id, anomaly_to_id):
        self.category_to_id = category_to_id
        self.anomaly_to_id = anomaly_to_id
    
    def calculate_high_reward(self, action_high, label):
        # action_high is the predicted category ID
        # label is the true category ID
        if action_high == label:
            return 1
        else:
            return -1
    
    def calculate_low_reward(self, action_low, label):
        # action_low is the predicted anomaly ID
        # label is the true anomaly ID
        true_anomaly = label  # Assuming label corresponds to anomaly ID
        if action_low == true_anomaly:
            return 1
        else:
            return -1
