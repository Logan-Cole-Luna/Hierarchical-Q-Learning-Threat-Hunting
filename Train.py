# train.py

"""
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
import logging
import argparse
import torch
import random
from agents.high_level_agent import HighLevelAgent
from agents.low_level_agent import LowLevelAgent
from utils.q_network import QNetwork
from utils.intrusion_detection_env import IntrusionDetectionEnv
from utils.trainer import Trainer
from scripts.evaluator import Evaluator

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_agents(overfit=False):
    set_seed(42)
    data_dir = 'data'
    mappings_dir = os.path.join(data_dir, 'mappings')
    category_to_id_path = os.path.join(mappings_dir, 'category_to_id.json')
    anomaly_to_id_path = os.path.join(mappings_dir, 'anomaly_to_id.json')

    if not os.path.exists(category_to_id_path) or not os.path.exists(anomaly_to_id_path):
        logging.error("Mapping files not found. Please run preprocess.py first.")
        sys.exit(1)

    if overfit:
        logging.info("Training in overfit mode using subset of data.")
        X_high_train = np.load(os.path.join(data_dir, 'subset', 'X_high_subset.npy'))
        X_low_train = np.load(os.path.join(data_dir, 'subset', 'X_low_subset.npy'))
        y_high_train = np.load(os.path.join(data_dir, 'subset', 'y_high_subset.npy'))
        y_low_train = np.load(os.path.join(data_dir, 'subset', 'y_low_subset.npy'))
        X_high_test = X_high_train.copy()
        X_low_test = X_low_train.copy()
        y_high_test = y_high_train.copy()
        y_low_test = y_low_train.copy()
    else:
        logging.info("Training in normal mode using full dataset.")
        X_high_train = np.load(os.path.join(data_dir, 'X_high_train.npy'))
        X_low_train = np.load(os.path.join(data_dir, 'X_low_train.npy'))
        y_high_train = np.load(os.path.join(data_dir, 'y_high_train.npy'))
        y_low_train = np.load(os.path.join(data_dir, 'y_low_train.npy'))
        X_high_test = np.load(os.path.join(data_dir, 'X_high_test.npy'))
        X_low_test = np.load(os.path.join(data_dir, 'X_low_test.npy'))
        y_high_test = np.load(os.path.join(data_dir, 'y_high_test.npy'))
        y_low_test = np.load(os.path.join(data_dir, 'y_low_test.npy'))

    # Initialize environment without historical states
    env = IntrusionDetectionEnv(
        X_high=X_high_train,
        X_low=X_low_train,
        y_high=y_high_train,
        y_low=y_low_train,
        high_agent_action_space=None,
        low_agent_action_space=None,
        mappings_dir=mappings_dir,
        history_length=1  # Remove temporal stacking
    )

    # Define action space sizes
    high_agent_action_space = len(env.category_to_id)
    low_agent_action_space = len(env.anomaly_to_id)
    env.high_agent_action_space = high_agent_action_space
    env.low_agent_action_space = low_agent_action_space

    # Calculate the correct state sizes without extra features
    feature_size_high = X_high_train.shape[1]
    feature_size_low = X_low_train.shape[1]
    state_size_high = feature_size_high  # No history stacking
    state_size_low = feature_size_low    # No history stacking

    # Initialize agents with correct state sizes
    high_agent = HighLevelAgent(
        state_size=state_size_high,
        action_size=high_agent_action_space,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=10  # Now supported by BaseAgent
    )

    low_agent = LowLevelAgent(
        state_size=state_size_low,
        action_size=low_agent_action_space,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=10  # Now supported by BaseAgent
    )

    # Initialize Trainer
    trainer = Trainer(
        env=env,
        high_agent=high_agent,
        low_agent=low_agent,
        episodes=1000
    )

    # Train agents
    trainer.train()

    # Save models
    high_model_path = 'high_level_agent_overfit.pth' if overfit else 'high_level_agent.pth'
    low_model_path = 'low_level_agent_overfit.pth' if overfit else 'low_level_agent.pth'
    if overfit:
        logging.info("Training completed in overfit mode. Saving overfit models.")
    else:
        logging.info("Training completed. Saving models.")

    high_agent.save(high_model_path)
    low_agent.save(low_model_path)
    logging.info(f"Models saved as {high_model_path} and {low_model_path}.")

    # Prepare test data without historical states
    X_high_test_historical = X_high_test  # No stacking
    X_low_test_historical = X_low_test    # No stacking
    y_high_test_historical = y_high_test
    y_low_test_historical = y_low_test

    # Debugging: Print shapes
    print(f"Shape of X_high_test_historical: {X_high_test_historical.shape}")
    print(f"Shape of X_low_test_historical: {X_low_test_historical.shape}")
    print(f"Shape of y_high_test_historical: {y_high_test_historical.shape}")
    print(f"Shape of y_low_test_historical: {y_low_test_historical.shape}")

    # Initialize Evaluator with updated parameters
    evaluator = Evaluator(
        high_agent=high_agent,
        low_agent=low_agent,
        X_high_test=X_high_test_historical,
        X_low_test=X_low_test_historical,
        y_high_test=y_high_test_historical,
        y_low_test=y_low_test_historical
    )

    # Evaluate agents
    evaluator.evaluate_agents()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Basic Q-Learning Threat Hunting Training')
    parser.add_argument('--overfit', action='store_true', help='Train in overfit mode using a subset of data')
    args = parser.parse_args()
    train_agents(overfit=args.overfit)
