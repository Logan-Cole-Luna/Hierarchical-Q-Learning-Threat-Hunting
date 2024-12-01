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
    y_pred = []
    y_scores = []
    
    # Batch processing
    logger.info("Starting batch prediction for Binary Classifier...")
    for start in range(0, len(X_test), batch_size):
        end = start + batch_size
        X_batch = X_test[start:end]
        preds, scores = binary_agent.predict_batch(X_batch)
        y_pred.extend(preds)
        y_scores.extend(scores[:, 1])  # Assuming class '1' is 'Malicious'
    logger.info("Batch prediction completed.")
    
    classes = list(binary_agent.label_dict.keys())
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    cm_normalized = confusion_matrix(y_test_encoded, y_pred, normalize='true')
    
    if save_confusion_matrix:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(6,5))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix (Normalized)')
        cm_path = os.path.join(save_path, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Confusion matrix saved to {cm_path}")
    
    # Generate Classification Report
    report = classification_report(y_test_encoded, y_pred, target_names=classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = os.path.join(save_path, 'binary_classification_report.csv')
    report_df.to_csv(report_csv_path)
    logger.info(f"Binary Classification report saved to {report_csv_path}")
    
    # Generate ROC Curve
    if save_roc_curve:
        fpr, tpr, _ = roc_curve(y_test_encoded, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        roc_path = os.path.join(save_path, 'roc_curve.png')
        plt.savefig(roc_path)
        plt.close()
        logger.info(f"ROC curve saved to {roc_path}")
    
    return report

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix as a heatmap with actual label names."""
    plt.figure(figsize=(12,10))
    
    # Create a DataFrame for better labeling
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    save_file = os.path.join(save_path, 'rl_confusion_matrix.png')
    plt.savefig(save_file, bbox_inches='tight')
    plt.close()

def plot_roc_curves(fpr, tpr, roc_auc, class_names, save_path):
    """
    Plot and save ROC curves for multiple classes.
    
    Parameters:
    - fpr (dict): False positive rates for each class 
    - tpr (dict): True positive rates for each class
    - roc_auc (dict): ROC AUC scores for each class
    - class_names (list): List of class names
    - save_path (str): Directory to save the plot
    """
    plt.figure(figsize=(10,8))
    
    # Plot ROC curve for each class
    for i, class_name in enumerate(class_names):
        if i in fpr and i in tpr and i in roc_auc:
            plt.plot(fpr[i], tpr[i],
                    label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    
    # Save the plot
    save_file = os.path.join(save_path, 'rl_roc_curves.png')
    plt.savefig(save_file)
    plt.close()

def evaluate_rl_agent(agent, env, label_dict, multi_test_df, batch_size=1, save_confusion_matrix=True, save_roc_curves=True, save_path='results'):
    """Evaluate a trained RL agent with human-readable labels."""
    logger.info("Starting evaluation of RL Agent...")
    y_true = []
    y_scores = []
    
    # Reset environment
    current_states, current_labels = env.reset()
    done = False

    while not done:
        # Get state and true label
        if current_states is None or len(current_states) == 0:
            break

        # Get action probabilities from agent
        state_tensor = torch.FloatTensor(current_states).to(agent.device)
        action_probs = agent.get_action_probabilities(state_tensor)

        # Store predictions and true labels
        y_true.extend(current_labels)
        y_scores.extend(action_probs.detach().cpu().numpy())

        # Take action in environment
        actions = np.argmax(action_probs.detach().cpu().numpy(), axis=1)
        current_states, rewards, done, current_labels = env.step(actions)

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Ensure we have predictions
    if len(y_true) == 0 or len(y_scores) == 0:
        logger.error("No predictions generated during evaluation")
        return None

    # Generate confusion matrix
    y_pred = np.argmax(y_scores, axis=1)
    
    # Convert numeric predictions to label names
    inv_label_dict = {v: k for k, v in label_dict.items()}
    y_true_labels = np.array([inv_label_dict[y] for y in y_true])
    y_pred_labels = np.array([inv_label_dict[y] for y in y_pred])
    
    # Generate confusion matrix with actual labels
    if save_confusion_matrix:
        cm = confusion_matrix(y_true_labels, y_pred_labels)
        class_names = list(label_dict.keys())  # Use actual label names
        plot_confusion_matrix(cm, class_names, save_path)
        logger.info(f"RL Confusion matrix saved to {save_path}/rl_confusion_matrix.png")

    # Calculate ROC curves only if we have enough samples
    if save_roc_curves and len(y_true) > 1:
        try:
            # One-hot encode true labels
            y_true_binarized = label_binarize(y_true, classes=range(len(label_dict)))
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(len(label_dict)):
                if i in y_true:  # Only calculate for classes that exist in y_true
                    fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_scores[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
            
            plot_roc_curves(fpr, tpr, roc_auc, list(label_dict.keys()), save_path)
            
        except Exception as e:
            logger.warning(f"Could not generate ROC curves: {str(e)}")

    # Generate classification report with actual labels
    clf_report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
    
    # Save classification report
    report_path = os.path.join(save_path, 'rl_classification_report.csv')
    pd.DataFrame(clf_report).transpose().to_csv(report_path)
    logger.info(f"RL Classification report saved to {report_path}")

    return {
        'y_true': y_true_labels,  # Return actual labels instead of numbers
        'y_pred': y_pred_labels,
        'y_scores': y_scores,
        'classification_report': clf_report
    }

