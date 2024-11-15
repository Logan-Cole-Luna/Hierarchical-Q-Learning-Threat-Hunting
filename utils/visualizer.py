# visuals.py

"""
Visualization Utilities for Hierarchical Q-Learning Threat Hunting

This module provides a `Visuals` class for plotting training loss, confusion matrices, and ROC curves.
Used for evaluating and visualizing the performance of Q-Networks.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import numpy as np

class Visualizer:
    """
    A class containing visualization methods for training and evaluating Q-Networks.
    """

    @staticmethod
    def plot_loss(loss_high_history, loss_low_history):
        """
        Plots training loss for high-level and low-level Q-Networks.

        Parameters:
        -----------
        loss_high_history : list
            List of high-level Q-Network loss values.
        loss_low_history : list
            List of low-level Q-Network loss values.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(loss_high_history) + 1), loss_high_history, label='High-Level Loss')
        plt.plot(range(1, len(loss_low_history) + 1), loss_low_history, label='Low-Level Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, title, target_names, cmap='Blues'):
        """
        Plots a confusion matrix for model predictions.

        Parameters:
        -----------
        cm : ndarray
            Confusion matrix array.
        title : str
            Title for the confusion matrix plot.
        target_names : list
            List of class names for labeling the matrix.
        cmap : str, optional
            Color map for the matrix (default is 'Blues').
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    @staticmethod
    def plot_roc_curves(predictions, y_binarized, unique_labels, title):
        """
        Plots ROC curves for each class in a multi-class classification.

        Parameters:
        -----------
        predictions : ndarray
            Prediction probabilities for each class.
        y_binarized : ndarray
            Binarized ground truth labels.
        unique_labels : list
            List of unique label names for the classes.
        title : str
            Title for the ROC plot.
        """
        n_classes = y_binarized.shape[1]
        fpr, tpr, roc_auc = {}, {}, {}
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 8))
        colors = ['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'red']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of Class {unique_labels[i]} (area = {roc_auc[i]:0.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()
