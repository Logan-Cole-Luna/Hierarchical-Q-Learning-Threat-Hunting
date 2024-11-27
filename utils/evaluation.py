# utils/evaluation.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
import json
import os
import logging

logger = logging.getLogger(__name__)

def evaluate_binary_classifier(binary_agent, binary_test_df, batch_size=256, save_confusion_matrix=True, save_roc_curve=True, save_path='results/binary_classification'):
    """
    Evaluates the binary classifier on the test dataset and generates relevant visuals.

    Parameters:
    - binary_agent (BinaryAgent): Trained binary classifier agent.
    - binary_test_df (pd.DataFrame): Test dataset for binary classification.
    - batch_size (int): Number of samples per batch for prediction.
    - save_confusion_matrix (bool): Whether to save the confusion matrix plot.
    - save_roc_curve (bool): Whether to save the ROC curve plot.
    - save_path (str): Directory to save evaluation results.

    Returns:
    - report (dict): Classification report as a dictionary.
    """
    os.makedirs(save_path, exist_ok=True)
    
    X_test = binary_test_df[binary_agent.feature_cols].values
    y_test = binary_test_df['Threat']
    y_test_encoded = y_test.map(binary_agent.label_dict).astype(int).values
    
    # Initialize predictions array
    y_pred = np.empty_like(y_test_encoded)
    y_scores = np.empty((len(y_test_encoded), len(binary_agent.label_dict)), dtype=np.float32)
    
    # Batch processing
    logger.info("Starting batch prediction for Binary Classifier...")
    for start in range(0, len(X_test), batch_size):
        end = start + batch_size
        X_batch = X_test[start:end]
        preds, scores = binary_agent.predict_batch(X_batch)
        y_pred[start:end] = preds
        y_scores[start:end] = scores
    logger.info("Batch prediction completed.")
    
    classes = list(binary_agent.label_dict.keys())
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    cm_normalized = confusion_matrix(y_test_encoded, y_pred, normalize='true')
    
    if save_confusion_matrix:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", xticklabels=classes, yticklabels=classes, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix - Binary Classification')
        plt.tight_layout()
        cm_path = os.path.join(save_path, 'binary_confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Binary Confusion matrix saved to {cm_path}")
    
    # Generate Classification Report
    report = classification_report(y_test_encoded, y_pred, target_names=classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = os.path.join(save_path, 'binary_classification_report.csv')
    report_df.to_csv(report_csv_path)
    logger.info(f"Binary Classification report saved to {report_csv_path}")
    
    # Generate ROC Curve
    if save_roc_curve:
        # Binarize the labels for ROC
        y_true_binarized = label_binarize(y_test_encoded, classes=list(binary_agent.label_dict.values()))
        if y_true_binarized.shape[1] == 1:
            y_true_binarized = np.hstack([1 - y_true_binarized, y_true_binarized])
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        plt.figure(figsize=(8, 6))
        for i in range(len(classes)):
            plt.plot(fpr[i], tpr[i], label=f'ROC curve for {classes[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - Binary Classification')
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(save_path, 'binary_roc_curve.png')
        plt.savefig(roc_path)
        plt.close()
        logger.info(f"Binary ROC curve saved to {roc_path}")
    
    return report

def evaluate_rl_agent(agent, env, label_dict, multi_test_df, batch_size=256, save_confusion_matrix=True, save_roc_curves=True, save_path='results/multi_class_classification'):
    """
    Evaluates the trained RL agent on the multi-class test dataset and generates relevant visuals.

    Parameters:
    - agent (Agent): Trained RL agent.
    - env (NetworkClassificationEnv): Evaluation environment.
    - label_dict (dict): Dictionary mapping labels to indices.
    - multi_test_df (pd.DataFrame): Test dataset for multi-class classification.
    - batch_size (int): Number of samples per batch for prediction.
    - save_confusion_matrix (bool): Whether to save the confusion matrix plot.
    - save_roc_curves (bool): Whether to save ROC curves.
    - save_path (str): Directory to save evaluation results.

    Returns:
    - metrics (dict): Dictionary containing evaluation metrics.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Reset environment
    states, labels = env.reset()
    all_actions = []
    all_labels = []
    done = False

    # Initialize predictions array
    y_pred = []
    y_scores = []

    logger.info("Starting evaluation of RL Agent...")
    while not done:
        actions = agent.act(states)
        next_states, rewards, done, next_labels = env.step(actions)
        all_actions.extend(actions)
        all_labels.extend(labels)
        states = next_states
        labels = next_labels

    y_pred = np.array(all_actions)
    y_true = np.array(all_labels)
    
    # Assuming that RL agent provides action probabilities or Q-values for ROC
    # For this example, we'll simulate probabilities using one-hot encoding
    y_scores = np.zeros((len(y_true), len(label_dict)))
    y_scores[np.arange(len(y_true)), y_pred] = 1.0  # Simplistic approach

    classes = list(label_dict.keys())
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    
    if save_confusion_matrix:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", xticklabels=classes, yticklabels=classes, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix - RL Agent')
        plt.tight_layout()
        cm_path = os.path.join(save_path, 'rl_confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"RL Confusion matrix saved to {cm_path}")
    
    # Generate Classification Report
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = os.path.join(save_path, 'rl_classification_report.csv')
    report_df.to_csv(report_csv_path)
    logger.info(f"RL Classification report saved to {report_csv_path}")
    
    # Generate ROC Curves
    if save_roc_curves:
        y_true_binarized = label_binarize(y_true, classes=list(label_dict.values()))
        if y_true_binarized.shape[1] == 1:
            y_true_binarized = np.hstack([1 - y_true_binarized, y_true_binarized])
        
        n_classes = y_true_binarized.shape[1]
    
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    
        # Plot ROC curves
        plt.figure(figsize=(12, 10))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'ROC curve for {classes[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - RL Agent')
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(save_path, 'rl_roc_curves.png')
        plt.savefig(roc_path)
        plt.close()
        logger.info(f"RL ROC curves saved to {roc_path}")
    
    # Compile metrics
    metrics = {
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'roc_auc': roc_auc
    }
    
    return metrics
