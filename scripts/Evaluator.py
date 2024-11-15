# evaluator.py

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import warnings
from utils.visualizer import Visuals  # Import the Visuals class
from sklearn.preprocessing import label_binarize

class Evaluator:
    def __init__(self, high_agent, low_agent, X_high_test, X_low_test, y_high_test, y_low_test):
        """
        Initializes the Evaluator with trained agents and test data.

        Parameters:
        -----------
        high_agent : HighLevelAgent
            Trained high-level agent.
        low_agent : LowLevelAgent
            Trained low-level agent.
        X_high_test : np.ndarray
            Test feature matrix for high-level actions.
        X_low_test : np.ndarray
            Test feature matrix for low-level actions.
        y_high_test : np.ndarray
            True labels for high-level test data.
        y_low_test : np.ndarray
            True labels for low-level test data.
        """
        self.high_agent = high_agent
        self.low_agent = low_agent
        self.X_high_test = X_high_test
        self.X_low_test = X_low_test
        self.y_high_test = y_high_test
        self.y_low_test = y_low_test

    def evaluate_agents(self):
        """
        Evaluates both high-level and low-level agents on the test data, prints classification reports,
        and generates visualizations.
        """
        # High-Level Agent Evaluation
        high_pred_q = self.high_agent.q_network.predict(self.X_high_test)
        high_pred_labels = np.argmax(high_pred_q, axis=1)

        print("High-Level Agent Evaluation:")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(classification_report(
                self.y_high_test,
                high_pred_labels,
                zero_division=0
            ))

        # Low-Level Agent Evaluation
        low_pred_q = self.low_agent.q_network.predict(self.X_low_test)
        low_pred_labels = np.argmax(low_pred_q, axis=1)

        print("\nLow-Level Agent Evaluation:")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(classification_report(
                self.y_low_test,
                low_pred_labels,
                zero_division=0
            ))

        # Confusion Matrices
        cm_high = confusion_matrix(self.y_high_test, high_pred_labels)
        cm_low = confusion_matrix(self.y_low_test, low_pred_labels)

        # Plot confusion matrices using Visuals class
        unique_high_labels = np.unique(self.y_high_test)
        high_target_names = [f"Class {label}" for label in unique_high_labels]
        Visuals.plot_confusion_matrix(cm_high, 'Confusion Matrix - High-Level Agent', high_target_names, cmap='Blues')

        unique_low_labels = np.unique(self.y_low_test)
        low_target_names = [f"Class {label}" for label in unique_low_labels]
        Visuals.plot_confusion_matrix(cm_low, 'Confusion Matrix - Low-Level Agent', low_target_names, cmap='Greens')

        # ROC Curves and AUC
        # Binarize the labels for ROC computation
        y_high_binarized = label_binarize(self.y_high_test, classes=unique_high_labels)
        y_low_binarized = label_binarize(self.y_low_test, classes=unique_low_labels)

        # Ensure that predictions are in probability format for ROC curves
        high_pred_proba = self._convert_q_values_to_probabilities(high_pred_q)
        low_pred_proba = self._convert_q_values_to_probabilities(low_pred_q)

        # Plot ROC curves using Visuals class
        Visuals.plot_roc_curves(high_pred_proba, y_high_binarized, unique_high_labels, 'ROC Curves - High-Level Agent')
        Visuals.plot_roc_curves(low_pred_proba, y_low_binarized, unique_low_labels, 'ROC Curves - Low-Level Agent')

    def _convert_q_values_to_probabilities(self, q_values):
        """
        Converts Q-values to probabilities using softmax for ROC curve computation.

        Parameters:
        -----------
        q_values : np.ndarray
            Q-values output from the agent's network.

        Returns:
        --------
        np.ndarray
            Probabilities corresponding to each class.
        """
        # Avoid overflow by subtracting the max Q-value
        q_values_exp = np.exp(q_values - np.max(q_values, axis=1, keepdims=True))
        probabilities = q_values_exp / np.sum(q_values_exp, axis=1, keepdims=True)
        return probabilities
