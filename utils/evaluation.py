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
import onnxruntime as ort

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
    y_test_encoded = y_test.map(binary_agent.label_mapping).astype(int).values
    
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
    
    classes = list(binary_agent.label_mapping.keys())
    
    # Check for prediction alignment
    y_pred_array = np.array(y_pred)
    y_test_array = np.array(y_test_encoded)
    if np.array_equal(y_pred_array, y_test_array):
        logger.warning("Predictions perfectly match the true labels. Check for data leakage.")
    
    # Add data quality checks
    logger.info("\nData Quality Checks:")
    logger.info(f"Total samples: {len(binary_test_df)}")
    logger.info(f"Class distribution:\n{binary_test_df['Threat'].value_counts(normalize=True).round(4) * 100}%")
    
    # Check for potential duplicates
    duplicates = binary_test_df[binary_agent.feature_cols].duplicated().sum()
    logger.info(f"Number of duplicate feature vectors: {duplicates}")
    
    # Check for extreme feature values
    feature_stats = binary_test_df[binary_agent.feature_cols].describe()
    logger.info("\nFeature Statistics:")
    logger.info(feature_stats)
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    cm_normalized = confusion_matrix(y_test_encoded, y_pred, normalize='true')
    
    if save_confusion_matrix:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10,8))
        cm_normalized = confusion_matrix(y_test_encoded, y_pred, normalize='true')
        cm_percentage = cm_normalized * 100  # Convert to percentages
        
        sns.heatmap(cm_percentage, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues', 
                   xticklabels=classes, 
                   yticklabels=classes)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix (Normalized %)')
        
        # Add text annotations for actual counts
        cm_counts = confusion_matrix(y_test_encoded, y_pred)
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j + 0.2, i + 0.7, f'(n={cm_counts[i,j]})', 
                        color='black', fontsize=9)
        
        cm_path = os.path.join(save_path, 'confusion_matrix.png')
        plt.tight_layout()
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
    
    # Normalize the confusion matrix to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create a DataFrame for better labeling
    cm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    
    # Plot heatmap with both percentages and raw counts
    sns.heatmap(cm_df, 
                annot=True, 
                fmt='.1f', 
                cmap='Blues',
                vmin=0, 
                vmax=100)
    
    # Add raw counts as text
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j + 0.2, i + 0.7, f'(n={cm[i,j]})', 
                    fontsize=8, color='black')
    
    plt.title('Confusion Matrix\n(percentages with raw counts)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    save_file = os.path.join(save_path, 'rl_confusion_matrix.png')
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
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

def evaluate_rl_agent(session, test_df, feature_cols, label_col, label_mapping, batch_size=256, save_confusion_matrix=True, save_roc_curve=True, save_path='results/multi_class_classification'):
    """
    Evaluates the RL agent using the provided ONNX model.

    Parameters:
    - session (onnxruntime.InferenceSession): The ONNX inference session for the RL model.
    - test_df (pd.DataFrame): Test dataset for multi-class classification.
    - feature_cols (list): List of feature column names.
    - label_col (str): Name of the label column.
    - label_mapping (dict): Mapping from label names to integers.
    - batch_size (int): Number of samples per batch for prediction.
    - save_confusion_matrix (bool): Whether to save the confusion matrix plot.
    - save_roc_curve (bool): Whether to save the ROC curve plot.
    - save_path (str): Directory to save evaluation results.

    Returns:
    - report (dict): Classification report as a dictionary.
    """
    os.makedirs(save_path, exist_ok=True)
    
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[label_col]
    y_test_encoded = y_test.map(label_mapping).astype(int).values
    
    # Initialize predictions array
    y_pred = []
    y_scores = []
    
    # Batch processing
    logger.info("Starting batch prediction for RL Agent...")
    for start in range(0, len(X_test), batch_size):
        end = start + batch_size
        X_batch = X_test[start:end]
        inputs = {session.get_inputs()[0].name: X_batch}
        outputs = session.run(None, inputs)
        y_pred_batch = np.argmax(outputs[0], axis=1)
        y_pred.extend(y_pred_batch)
        y_scores.extend(outputs[0])  # Assuming outputs[0] contains raw scores or probabilities
    logger.info("Batch prediction completed.")
    
    classes = list(label_mapping.keys())
    
    # Map predictions back to original labels
    y_pred_labels = [classes[pred] for pred in y_pred]
    
    # Generate Classification Report
    report = classification_report(y_test_encoded, y_pred, target_names=classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = os.path.join(save_path, 'rl_classification_report.csv')
    report_df.to_csv(report_csv_path)
    logger.info(f"RL Classification report saved to {report_csv_path}")
    
    # Return metrics in the expected structure
    metrics = {
        'accuracy': report['accuracy'],
        'per_class': {}
    }
    
    for class_name in classes:
        if class_name in report:
            metrics['per_class'][class_name] = {
                'precision': report[class_name]['precision'],
                'recall': report[class_name]['recall'],
                'f1-score': report[class_name]['f1-score']
            }
    
    return metrics
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    cm_normalized = confusion_matrix(y_test_encoded, y_pred, normalize='true')
    
    if save_confusion_matrix:
        plt.figure(figsize=(10,8))
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues', 
                   xticklabels=classes, 
                   yticklabels=classes)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('RL Confusion Matrix (Normalized %)')

        # Add text annotations for actual counts
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j + 0.2, i + 0.7, f'(n={cm[i,j]})', 
                        color='black', fontsize=9)
        
        cm_path = os.path.join(save_path, 'rl_confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"RL Confusion matrix saved to {cm_path}")
    
    # Generate ROC Curve (if applicable)
    if save_roc_curve:
        try:
            # Binarize the output
            y_test_binarized = label_binarize(y_test_encoded, classes=range(len(label_mapping)))
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(len(label_mapping)):
                fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot ROC curves
            plt.figure(figsize=(10,8))
            for i, class_name in enumerate(classes):
                plt.plot(fpr[i], tpr[i], label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('RL Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            
            roc_path = os.path.join(save_path, 'rl_roc_curve.png')
            plt.savefig(roc_path)
            plt.close()
            logger.info(f"RL ROC curve saved to {roc_path}")
        except Exception as e:
            logger.warning(f"Could not generate ROC curves for RL Agent: {str(e)}")
    
    return report

