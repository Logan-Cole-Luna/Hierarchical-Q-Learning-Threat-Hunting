"""
Train.py

Main script for setting up, training, evaluating, and overfit testing high-level and low-level agents
in an intrusion detection environment. The script loads preprocessed data, initializes the environment 
and agents, trains the agents on full data, tests for overfitting on a subset, saves their models, 
and performs evaluation.

Modules:
    - main: Orchestrates the entire training, evaluation, and overfitting testing pipeline.

Functions:
    - set_seed: Sets random seeds for reproducibility.
    - train_and_evaluate: Trains agents, performs evaluation, and saves models.
    - test_for_overfit: Tests agents on subset data for overfitting.
    - main: Calls training, overfitting test, and evaluation.
"""

import os
import sys
import numpy as np
from IntrusionDetectionEnv import IntrusionDetectionEnv
from agents.HighLevelAgent import HighLevelAgent
from agents.LowLevelAgent import LowLevelAgent
from RewardCalculator import RewardCalculator
from Trainer import Trainer
from scripts.Evaluator import Evaluator
from QNetwork import QNetwork
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import logging
from Visuals import Visuals

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed=42):
    import torch
    import random

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_and_evaluate():
    """
    Function for setting up, training, and evaluating agents on full data.
    """
    # Load data and initialize mappings
    mappings_dir = os.path.join('data', 'mappings')
    category_to_id_path = os.path.join(mappings_dir, 'category_to_id.json')
    anomaly_to_id_path = os.path.join(mappings_dir, 'anomaly_to_id.json')
    
    if not os.path.exists(category_to_id_path) or not os.path.exists(anomaly_to_id_path):
        print("Error: Mapping files not found. Please run preprocess.py first.")
        sys.exit(1)
    
    # Load training and test data
    data_dir = 'data'
    X_high_train = np.load(os.path.join(data_dir, 'X_high_train.npy'))
    X_low_train = np.load(os.path.join(data_dir, 'X_low_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_high_test = np.load(os.path.join(data_dir, 'X_high_test.npy'))
    X_low_test = np.load(os.path.join(data_dir, 'X_low_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Initialize environment
    env = IntrusionDetectionEnv(
        X_high=X_high_train,
        X_low=X_low_train,
        y=y_train,
        high_agent_action_space=None,
        low_agent_action_space=None,
        mappings_dir=mappings_dir
    )
    
    # Define action space sizes
    high_agent_action_space = len(env.category_to_id)
    low_agent_action_space = len(env.anomaly_to_id)
    env.high_agent_action_space = high_agent_action_space
    env.low_agent_action_space = low_agent_action_space
    
    # Initialize agents
    high_agent = HighLevelAgent(
        state_size=X_high_train.shape[1],
        action_size=high_agent_action_space,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=64
    )
    
    low_agent = LowLevelAgent(
        state_size=X_low_train.shape[1],
        action_size=low_agent_action_space,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=64
    )
    
    # Train agents
    trainer = Trainer(
        env=env,
        high_agent=high_agent,
        low_agent=low_agent,
        episodes=1000
    )
    trainer.train()
    
    # Save models
    high_agent.save('high_level_agent.h5')
    low_agent.save('low_level_agent.h5')
    print("Training completed and models saved.")
    
    # Evaluation
    evaluator = Evaluator(
        high_agent=high_agent,
        low_agent=low_agent,
        X_high_test=X_high_test,
        X_low_test=X_low_test,
        y_test=y_test
    )
    evaluator.evaluate_agents()

def test_for_overfit():
    """
    Function to test for overfitting on a subset of data and visualize results.
    """
    set_seed(42)

    # Load subset data
    X_high_subset = np.load('./data/subset/X_high_subset.npy')
    X_low_subset = np.load('./data/subset/X_low_subset.npy')
    y_high_subset = np.load('./data/subset/y_high_subset.npy')
    y_low_subset = np.load('./data/subset/y_low_subset.npy')

    # Initialize Q-Networks
    state_size_high = X_high_subset.shape[1]
    state_size_low = X_low_subset.shape[1]
    action_size = len(np.unique(y_high_subset))

    q_network_high = QNetwork(state_size_high, action_size, learning_rate=0.001)
    q_network_low = QNetwork(state_size_low, action_size, learning_rate=0.001)

    # Train with early stopping
    num_epochs = 1000
    patience = 100
    patience_counter = 0
    best_loss_high = float('inf')
    best_loss_low = float('inf')
    loss_high_history = []
    loss_low_history = []

    logging.info("Starting overfit testing on subset data...")

    for epoch in range(1, num_epochs + 1):
        loss_high = q_network_high.train_on_batch(X_high_subset, y_high_subset)
        loss_low = q_network_low.train_on_batch(X_low_subset, y_low_subset)

        loss_high_history.append(loss_high)
        loss_low_history.append(loss_low)

        if loss_high < best_loss_high and loss_low < best_loss_low:
            best_loss_high = loss_high
            best_loss_low = loss_low
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"No improvement in loss for {patience} epochs. Early stopping at epoch {epoch}.")
            break

    logging.info("Overfit testing completed.")

    # Visualize training loss
    Visuals.plot_loss(loss_high_history, loss_low_history)

    # Evaluation
    predictions_high = q_network_high.predict(X_high_subset)
    predictions_low = q_network_low.predict(X_low_subset)
    predicted_actions_high = np.argmax(predictions_high, axis=1)
    predicted_actions_low = np.argmax(predictions_low, axis=1)

    accuracy_high = np.mean(predicted_actions_high == y_high_subset)
    accuracy_low = np.mean(predicted_actions_low == y_low_subset)
    logging.info(f"High-Level Q-Network Accuracy on Subset: {accuracy_high * 100:.2f}%")
    logging.info(f"Low-Level Q-Network Accuracy on Subset: {accuracy_low * 100:.2f}%")

    target_names = [f"Class {label}" for label in np.unique(y_high_subset)]
    print("\nClassification Report for High-Level Q-Network:")
    print(classification_report(y_high_subset, predicted_actions_high, target_names=target_names, zero_division=0))
    print("Classification Report for Low-Level Q-Network:")
    print(classification_report(y_low_subset, predicted_actions_low, target_names=target_names, zero_division=0))

    cm_high = confusion_matrix(y_high_subset, predicted_actions_high)
    cm_low = confusion_matrix(y_low_subset, predicted_actions_low)
    Visuals.plot_confusion_matrix(cm_high, 'Confusion Matrix - High-Level Q-Network', target_names, cmap='Blues')
    Visuals.plot_confusion_matrix(cm_low, 'Confusion Matrix - Low-Level Q-Network', target_names, cmap='Greens')

    y_high_binarized = label_binarize(y_high_subset, classes=np.unique(y_high_subset))
    y_low_binarized = label_binarize(y_low_subset, classes=np.unique(y_low_subset))
    Visuals.plot_roc_curves(predictions_high, y_high_binarized, np.unique(y_high_subset), 'ROC Curves - High-Level Q-Network')
    Visuals.plot_roc_curves(predictions_low, y_low_binarized, np.unique(y_low_subset), 'ROC Curves - Low-Level Q-Network')

def main():
    train_and_evaluate()
    test_for_overfit()

if __name__ == "__main__":
    main()
