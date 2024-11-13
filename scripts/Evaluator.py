"""
Evaluator.py

Defines the `Evaluator` class, which evaluates the performance of high-level and low-level agents
using metrics such as classification reports and AUC-ROC scores. This module provides insights 
into the agents' prediction accuracy and model performance.

Classes:
    - Evaluator: Evaluates high-level and low-level agents on test data.
"""

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

class Evaluator:
    def __init__(self, high_agent, low_agent, X_high_test, X_low_test, y_test):
        """
        Initializes the Evaluator with test data and agents to evaluate.

        Parameters:
        -----------
        high_agent : HighLevelAgent
            The trained high-level agent.
        low_agent : LowLevelAgent
            The trained low-level agent.
        X_high_test : np.ndarray
            High-level test feature data.
        X_low_test : np.ndarray
            Low-level test feature data.
        y_test : np.ndarray
            True labels for test data.
        """
        self.high_agent = high_agent
        self.low_agent = low_agent
        self.X_high_test = X_high_test
        self.X_low_test = X_low_test
        self.y_test = y_test
    
    def evaluate_agents(self):
        """
        Evaluates the high-level and low-level agents on the test data using classification reports 
        and AUC-ROC scores. The function outputs performance metrics for each agent.
        """
        
        # Generate predictions for the high-level agent
        high_pred_q = self.high_agent.q_network.model.predict(self.X_high_test)
        high_pred_labels = np.argmax(high_pred_q, axis=1)
        
        # Generate predictions for the low-level agent
        low_pred_q = self.low_agent.q_network.model.predict(self.X_low_test)
        low_pred_labels = np.argmax(low_pred_q, axis=1)
        
        # Evaluate the high-level agent
        print("High-Level Agent Evaluation:")
        print(classification_report(self.y_test, high_pred_labels))
        
        # Calculate AUC-ROC score for the high-level agent
        try:
            auc_high = roc_auc_score(self.y_test, high_pred_q, multi_class='ovr')
            print(f"AUC-ROC High-Level: {auc_high:.4f}")
        except Exception as e:
            print(f"AUC-ROC High-Level: Cannot compute due to {e}")
        
        # Evaluate the low-level agent
        print("\nLow-Level Agent Evaluation:")
        print(classification_report(self.y_test, low_pred_labels))
        
        # Calculate AUC-ROC score for the low-level agent
        try:
            auc_low = roc_auc_score(self.y_test, low_pred_q, multi_class='ovr')
            print(f"AUC-ROC Low-Level: {auc_low:.4f}")
        except Exception as e:
            print(f"AUC-ROC Low-Level: Cannot compute due to {e}")
