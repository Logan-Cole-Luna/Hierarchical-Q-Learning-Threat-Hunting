"""
overfit.py

This script trains high-level and low-level Q-Networks on a small subset of data to test 
for overfitting. It includes training with early stopping, accuracy evaluation, and visualization 
of results such as confusion matrices and ROC curves.

Modules:
    - main: Executes training and evaluation of Q-Networks on subset data.

Functions:
    - set_seed: Sets random seeds for reproducibility across different libraries.
    - main: Loads subset data, trains Q-Networks, evaluates their performance, and visualizes results.
"""

import numpy as np
from QNetwork import QNetwork  # Ensure QNetwork.py is in the same directory or adjust the path
from sklearn.metrics import classification_report, confusion_matrix
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed=42):
    """
    Set random seeds for reproducibility across different libraries.

    Parameters:
    -----------
    seed : int, optional
        Seed value for random number generators (default is 42).
    """
    import torch
    import random
    import os

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    """
    Main function to train and evaluate Q-Networks on a subset of data for overfitting.
    
    Steps:
    ------
    1. Sets the seed for reproducibility.
    2. Loads subset data for high- and low-level features.
    3. Initializes high and low Q-Networks.
    4. Trains the networks with early stopping and logs the loss at intervals.
    5. Evaluates and prints classification metrics, plots confusion matrices, 
       and plots optional ROC curves for detailed performance analysis.
    """
    set_seed(42)  # For reproducibility

    # Load subset data
    X_high_subset = np.load('./data/subset/X_high_subset.npy')
    X_low_subset = np.load('./data/subset/X_low_subset.npy')
    y_subset = np.load('./data/subset/y_subset.npy')

    # Verify unique labels in the subset data
    unique_labels = np.unique(y_subset)
    logging.info(f"Unique labels in subset: {unique_labels}")

    # Initialize Q-Networks for high-level and low-level features
    state_size_high = X_high_subset.shape[1]
    state_size_low = X_low_subset.shape[1]
    action_size = len(unique_labels)  # Number of unique classes in subset

    q_network_high = QNetwork(state_size_high, action_size, learning_rate=0.001)
    q_network_low = QNetwork(state_size_low, action_size, learning_rate=0.001)

    # Training parameters
    num_epochs = 1000  # Total number of training epochs
    print_interval = 100  # Interval for logging progress
    patience = 100  # Early stopping patience
    best_loss_high = float('inf')
    best_loss_low = float('inf')
    patience_counter = 0

    # Lists to store loss history for plotting
    loss_high_history = []
    loss_low_history = []

    logging.info("Starting training...")

    for epoch in range(1, num_epochs + 1):
        # Train both Q-Networks on the subset data
        loss_high = q_network_high.train_on_batch(X_high_subset, y_subset)
        loss_low = q_network_low.train_on_batch(X_low_subset, y_subset)

        # Append losses to history for plotting
        loss_high_history.append(loss_high)
        loss_low_history.append(loss_low)

        # Logging training progress
        if epoch % print_interval == 0 or epoch == 1:
            logging.info(f"Epoch {epoch}/{num_epochs} - High-Level Loss: {loss_high:.4f}, Low-Level Loss: {loss_low:.4f}")

        # Early stopping check
        if loss_high < best_loss_high and loss_low < best_loss_low:
            best_loss_high = loss_high
            best_loss_low = loss_low
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping condition
        if patience_counter >= patience:
            logging.info(f"No improvement in loss for {patience} epochs. Early stopping at epoch {epoch}.")
            break

    logging.info("Training completed.")

    # Plot training loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_high_history) + 1), loss_high_history, label='High-Level Loss')
    plt.plot(range(1, len(loss_low_history) + 1), loss_low_history, label='Low-Level Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluation of High-Level Q-Network
    predictions_high = q_network_high.predict(X_high_subset)
    predicted_actions_high = np.argmax(predictions_high, axis=1)

    # Evaluation of Low-Level Q-Network
    predictions_low = q_network_low.predict(X_low_subset)
    predicted_actions_low = np.argmax(predictions_low, axis=1)

    # Calculate and log accuracy
    accuracy_high = np.mean(predicted_actions_high == y_subset)
    accuracy_low = np.mean(predicted_actions_low == y_subset)
    logging.info(f"High-Level Q-Network Accuracy on Subset: {accuracy_high * 100:.2f}%")
    logging.info(f"Low-Level Q-Network Accuracy on Subset: {accuracy_low * 100:.2f}%")

    # Detailed classification reports
    target_names = [f"Class {label}" for label in unique_labels]

    print("\nClassification Report for High-Level Q-Network:")
    print(classification_report(y_subset, predicted_actions_high, target_names=target_names, zero_division=0))

    print("Classification Report for Low-Level Q-Network:")
    print(classification_report(y_subset, predicted_actions_low, target_names=target_names, zero_division=0))

    # Plot confusion matrices
    cm_high = confusion_matrix(y_subset, predicted_actions_high)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_high, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - High-Level Q-Network')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    cm_low = confusion_matrix(y_subset, predicted_actions_low)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_low, annot=True, fmt='d', cmap='Greens',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Low-Level Q-Network')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Optional: ROC Curves for multi-class (requires label binarization)
    y_subset_binarized = label_binarize(y_subset, classes=unique_labels)
    n_classes = y_subset_binarized.shape[1]

    from sklearn.metrics import roc_curve, auc

    # ROC curves for High-Level Q-Network
    fpr_high = dict()
    tpr_high = dict()
    roc_auc_high = dict()
    for i in range(n_classes):
        fpr_high[i], tpr_high[i], _ = roc_curve(y_subset_binarized[:, i], predictions_high[:, i])
        roc_auc_high[i] = auc(fpr_high[i], tpr_high[i])

    # ROC curves for Low-Level Q-Network
    fpr_low = dict()
    tpr_low = dict()
    roc_auc_low = dict()
    for i in range(n_classes):
        fpr_low[i], tpr_low[i], _ = roc_curve(y_subset_binarized[:, i], predictions_low[:, i])
        roc_auc_low[i] = auc(fpr_low[i], tpr_low[i])

    # Plot ROC curves for High-Level Q-Network
    plt.figure(figsize=(10, 8))
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'red']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr_high[i], tpr_high[i], color=color, lw=2,
                 label=f'ROC curve of Class {unique_labels[i]} (area = {roc_auc_high[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - High-Level Q-Network')
    plt.legend(loc="lower right")
    plt.show()

    # Plot ROC curves for Low-Level Q-Network
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr_low[i], tpr_low[i], color=color, lw=2,
                 label=f'ROC curve of Class {unique_labels[i]} (area = {roc_auc_low[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Low-Level Q-Network')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()
