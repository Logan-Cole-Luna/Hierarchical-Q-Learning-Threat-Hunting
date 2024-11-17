# utils/visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os

def plot_training_metrics(reward_history, loss_history, save_path):
    plt.figure(figsize=(12,5))
    
    # Plot Rewards
    plt.subplot(1,2,1)
    plt.plot(np.arange(len(reward_history)), reward_history, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward by Episode')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1,2,2)
    plt.plot(np.arange(len(loss_history)), loss_history, label='Total Loss', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Total Loss')
    plt.title('Total Loss by Episode')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_path, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
