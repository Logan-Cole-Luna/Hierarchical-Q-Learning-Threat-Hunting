# overfit.py

"""
Hierarchical Q-Learning Threat Hunting - Overfit Test Script

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
from sklearn.preprocessing import label_binarize
from Visuals import Visuals  # Import Visuals class

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed=42):
    """
    Set random seeds for reproducibility across different libraries.
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
    """
    set_seed(42)

    # Load subset data
    X_high_subset = np.load('./data/subset/X_high_subset.npy')
    X_low_subset = np.load('./data/subset/X_low_subset.npy')
    y_high_subset = np.load('./data/subset/y_high_subset.npy')
    y_low_subset = np.load('./data/subset/y_low_subset.npy')

    # Verify unique labels in the subset data
    unique_labels = np.unique(y_high_subset)
    logging.info(f"Unique high-level labels in subset: {unique_labels}")

    # Initialize Q-Networks for high-level and low-level features
    state_size_high = X_high_subset.shape[1]
    state_size_low = X_low_subset.shape[1]
    action_size = len(unique_labels)

    q_network_high = QNetwork(state_size_high, action_size, learning_rate=0.001)
    q_network_low = QNetwork(state_size_low, action_size, learning_rate=0.001)

    # Training parameters
    num_epochs = 1000
    print_interval = 100
    patience = 100
    best_loss_high = float('inf')
    best_loss_low = float('inf')
    patience_counter = 0

    loss_high_history = []
    loss_low_history = []

    logging.info("Starting training...")

    for epoch in range(1, num_epochs + 1):
        loss_high = q_network_high.train_on_batch(X_high_subset, y_high_subset)
        loss_low = q_network_low.train_on_batch(X_low_subset, y_low_subset)

        loss_high_history.append(loss_high)
        loss_low_history.append(loss_low)

        if epoch % print_interval == 0 or epoch == 1:
            logging.info(f"Epoch {epoch}/{num_epochs} - High-Level Loss: {loss_high:.4f}, Low-Level Loss: {loss_low:.4f}")

        if loss_high < best_loss_high and loss_low < best_loss_low:
            best_loss_high = loss_high
            best_loss_low = loss_low
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"No improvement in loss for {patience} epochs. Early stopping at epoch {epoch}.")
            break

    logging.info("Training completed.")

    # Use Visuals to plot training loss
    Visuals.plot_loss(loss_high_history, loss_low_history)

    # Evaluation and Confusion Matrices
    predictions_high = q_network_high.predict(X_high_subset)
    predictions_low = q_network_low.predict(X_low_subset)
    predicted_actions_high = np.argmax(predictions_high, axis=1)
    predicted_actions_low = np.argmax(predictions_low, axis=1)

    accuracy_high = np.mean(predicted_actions_high == y_high_subset)
    accuracy_low = np.mean(predicted_actions_low == y_low_subset)
    logging.info(f"High-Level Q-Network Accuracy on Subset: {accuracy_high * 100:.2f}%")
    logging.info(f"Low-Level Q-Network Accuracy on Subset: {accuracy_low * 100:.2f}%")

    target_names = [f"Class {label}" for label in unique_labels]

    print("\nClassification Report for High-Level Q-Network:")
    print(classification_report(y_high_subset, predicted_actions_high, target_names=target_names, zero_division=0))

    print("Classification Report for Low-Level Q-Network:")
    print(classification_report(y_low_subset, predicted_actions_low, target_names=target_names, zero_division=0))

    # Plot confusion matrices using Visuals
    cm_high = confusion_matrix(y_high_subset, predicted_actions_high)
    cm_low = confusion_matrix(y_low_subset, predicted_actions_low)
    Visuals.plot_confusion_matrix(cm_high, 'Confusion Matrix - High-Level Q-Network', target_names, cmap='Blues')
    Visuals.plot_confusion_matrix(cm_low, 'Confusion Matrix - Low-Level Q-Network', target_names, cmap='Greens')

    # ROC Curves
    y_high_binarized = label_binarize(y_high_subset, classes=unique_labels)
    y_low_binarized = label_binarize(y_low_subset, classes=unique_labels)
    Visuals.plot_roc_curves(predictions_high, y_high_binarized, unique_labels, 'ROC Curves - High-Level Q-Network')
    Visuals.plot_roc_curves(predictions_low, y_low_binarized, unique_labels, 'ROC Curves - Low-Level Q-Network')

if __name__ == "__main__":
    main()
