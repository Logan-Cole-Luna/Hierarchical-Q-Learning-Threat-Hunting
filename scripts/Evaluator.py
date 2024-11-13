# scripts/Evaluator.py

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

class Evaluator:
    def __init__(self, high_agent, low_agent, X_high_test, X_low_test, y_test):
        self.high_agent = high_agent
        self.low_agent = low_agent
        self.X_high_test = X_high_test
        self.X_low_test = X_low_test
        self.y_test = y_test
    
    def evaluate_agents(self):
        # High-level agent predictions
        high_pred_q = self.high_agent.q_network.model.predict(self.X_high_test)
        high_pred_labels = np.argmax(high_pred_q, axis=1)
        
        # Low-level agent predictions
        low_pred_q = self.low_agent.q_network.model.predict(self.X_low_test)
        low_pred_labels = np.argmax(low_pred_q, axis=1)
        
        # Evaluate high-level agent
        print("High-Level Agent Evaluation:")
        print(classification_report(self.y_test, high_pred_labels))
        try:
            auc_high = roc_auc_score(self.y_test, high_pred_q, multi_class='ovr')
            print(f"AUC-ROC High-Level: {auc_high:.4f}")
        except Exception as e:
            print(f"AUC-ROC High-Level: Cannot compute due to {e}")
        
        # Evaluate low-level agent
        print("\nLow-Level Agent Evaluation:")
        print(classification_report(self.y_test, low_pred_labels))
        try:
            auc_low = roc_auc_score(self.y_test, low_pred_q, multi_class='ovr')
            print(f"AUC-ROC Low-Level: {auc_low:.4f}")
        except Exception as e:
            print(f"AUC-ROC Low-Level: Cannot compute due to {e}")
