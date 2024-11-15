# scripts/evaluator.py

import numpy as np
import torch
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import json
from utils.visualizer import Visuals
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class Evaluator:
    def __init__(self, high_agent, low_agent, X_high_test, X_low_test, y_high_test, y_low_test):
        self.high_agent = high_agent
        self.low_agent = low_agent
        self.X_high_test = X_high_test
        self.X_low_test = X_low_test
        self.y_high_test = y_high_test
        self.y_low_test = y_low_test

    def evaluate_agents(self):
        logging.info("Starting evaluation of agents.")

        # Convert high and low test features to tensors separately
        X_high_test_tensor = torch.FloatTensor(self.X_high_test).to(self.high_agent.device)
        X_low_test_tensor = torch.FloatTensor(self.X_low_test).to(self.low_agent.device)

        # High-Level Agent Predictions
        with torch.no_grad():
            high_q_values = self.high_agent.model(X_high_test_tensor)
            high_actions = torch.argmax(high_q_values, dim=1).cpu().numpy()
            high_probs = F.softmax(high_q_values, dim=1).cpu().numpy()

        # Low-Level Agent Predictions
        with torch.no_grad():
            low_q_values = self.low_agent.model(X_low_test_tensor)
            low_actions = torch.argmax(low_q_values, dim=1).cpu().numpy()
            low_probs = F.softmax(low_q_values, dim=1).cpu().numpy()

        # Evaluate High-Level Agent
        logging.info("Evaluating High-Level Agent:")
        self.evaluate_performance(
            true_labels=self.y_high_test,
            predicted_labels=high_actions,
            predicted_probs=high_probs,
            agent_type='High-Level'
        )

        # Evaluate Low-Level Agent
        logging.info("Evaluating Low-Level Agent:")
        self.evaluate_performance(
            true_labels=self.y_low_test,
            predicted_labels=low_actions,
            predicted_probs=low_probs,
            agent_type='Low-Level'
        )

    def evaluate_performance(self, true_labels, predicted_labels, predicted_probs, agent_type='Agent'):
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

        logging.info(f"{agent_type} Accuracy: {accuracy:.4f}")
        logging.info(f"{agent_type} Precision: {precision:.4f}")
        logging.info(f"{agent_type} Recall: {recall:.4f}")
        logging.info(f"{agent_type} F1-Score: {f1:.4f}")

        # Classification Report
        report = classification_report(true_labels, predicted_labels, zero_division=0)
        logging.info(f"{agent_type} Classification Report:\n{report}")

        # Confusion Matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        if agent_type == 'High-Level':
            with open('data/mappings/category_to_id.json', 'r') as f:
                classes = list(json.load(f).keys())
        else:
            with open('data/mappings/anomaly_to_id.json', 'r') as f:
                classes = list(json.load(f).keys())
        Visuals.plot_confusion_matrix(
            cm=cm,
            title=f'Normalized Confusion Matrix - {agent_type}',
            target_names=classes,
            save_path=f'results/confusion_matrix_{agent_type.lower()}.png'
        )
        logging.info(f"Confusion matrix plot saved to 'results/confusion_matrix_{agent_type.lower()}.png'.")

        # ROC Curves
        # Binarize the true labels for ROC computation
        n_classes = len(classes)
        y_binarized = label_binarize(true_labels, classes=list(range(n_classes)))
        Visuals.plot_roc_curves(
            predictions=predicted_probs,
            y_binarized=y_binarized,
            unique_labels=classes,
            title=f'ROC Curves - {agent_type}',
            save_path=f'results/roc_curves_{agent_type.lower()}.png'
        )
        logging.info(f"ROC curves plot saved to 'results/roc_curves_{agent_type.lower()}.png'.")

