# utils/visualizer.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import itertools
import os
import logging

logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save_path=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Parameters:
    - y_true (array-like): True labels
    - y_pred (array-like): Predicted labels
    - classes (list): List of class names
    - normalize (bool): Whether to apply normalization
    - title (str): Title of the plot
    - cmap: Color map
    - save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logger.info("Normalized confusion matrix")
    else:
        logger.info("Confusion matrix, without normalization")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Confusion matrix plot saved to {save_path}")
    plt.close()

def plot_training_metrics(reward_history, loss_history, save_path=None):
    """
    Plots the training reward and loss over episodes.
    
    Parameters:
    - reward_history (list): List of rewards per episode
    - loss_history (list): List of losses per episode
    - save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot Reward History
    plt.subplot(1, 2, 1)
    plt.plot(reward_history, label='Reward')
    plt.title('Training Reward Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # Plot Loss History
    plt.subplot(1, 2, 2)
    plt.plot(loss_history, label='Loss', color='orange')
    plt.title('Training Loss Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training metrics plot saved to {save_path}")
    plt.close()

def plot_roc_curves(fpr, tpr, roc_auc, classes, save_path=None):
    """
    Plots ROC curves for each class.
    
    Parameters:
    - fpr (dict): False Positive Rates for each class
    - tpr (dict): True Positive Rates for each class
    - roc_auc (dict): Area Under the Curve for each class
    - classes (list): List of class names
    - save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'ROC curve of class {class_name} (area = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"ROC curves plot saved to {save_path}")
    plt.close()
