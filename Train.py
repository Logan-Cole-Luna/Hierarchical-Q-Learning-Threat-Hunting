"""
Train.py

Main script for setting up, training, and evaluating high-level and low-level agents 
in an intrusion detection environment. The script loads preprocessed data, initializes 
the environment and agents, trains the agents, saves their models, and performs evaluation.

Modules:
    - main: Orchestrates the entire training and evaluation pipeline.

Functions:
    - main: Loads data, sets up the environment and agents, initiates training, 
            saves models, and evaluates the agents' performance.
"""

import sys
import os
import numpy as np
from IntrusionDetectionEnv import IntrusionDetectionEnv
from agents.HighLevelAgent import HighLevelAgent
from agents.LowLevelAgent import LowLevelAgent
from RewardCalculator import RewardCalculator
from Trainer import Trainer
from scripts.Evaluator import Evaluator

def main():
    """
    Main function for training and evaluating agents.
    
    Steps:
    ------
    1. Load category and anomaly mappings.
    2. Load preprocessed training and test data.
    3. Initialize the intrusion detection environment.
    4. Set action space sizes for high- and low-level agents.
    5. Initialize high-level and low-level agents with specified hyperparameters.
    6. Train agents using the Trainer class.
    7. Save the trained models.
    8. Initialize and run evaluations using the Evaluator class.
    """
    
    # Load mappings
    mappings_dir = os.path.join('data', 'mappings')
    category_to_id_path = os.path.join(mappings_dir, 'category_to_id.json')
    anomaly_to_id_path = os.path.join(mappings_dir, 'anomaly_to_id.json')
    
    # Ensure mapping files exist
    if not os.path.exists(category_to_id_path) or not os.path.exists(anomaly_to_id_path):
        print("Error: Mapping files not found. Please run preprocess.py first.")
        sys.exit(1)
    
    # Load preprocessed data for training and testing
    data_dir = 'data'  # Directory for data files
    X_high_train = np.load(os.path.join(data_dir, 'X_high_train.npy'))
    X_low_train = np.load(os.path.join(data_dir, 'X_low_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_high_test = np.load(os.path.join(data_dir, 'X_high_test.npy'))
    X_low_test = np.load(os.path.join(data_dir, 'X_low_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Initialize the environment with training data
    env = IntrusionDetectionEnv(
        X_high=X_high_train,
        X_low=X_low_train,
        y=y_train,
        high_agent_action_space=None,  # Set later based on mappings
        low_agent_action_space=None,   # Set later based on mappings
        mappings_dir=mappings_dir
    )
    
    # Define action space sizes from mappings
    high_agent_action_space = len(env.category_to_id)
    low_agent_action_space = len(env.anomaly_to_id)
    env.high_agent_action_space = high_agent_action_space
    env.low_agent_action_space = low_agent_action_space
    
    # Initialize high-level agent with specified parameters
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
    
    # Initialize low-level agent with specified parameters
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
    
    # Initialize Trainer for agent training
    trainer = Trainer(
        env=env,
        high_agent=high_agent,
        low_agent=low_agent,
        episodes=1000  # Total training episodes
    )
    
    # Start the training process
    trainer.train()
    
    # Save the trained models
    high_agent.save('high_level_agent.h5')
    low_agent.save('low_level_agent.h5')
    print("Training completed and models saved.")
    
    # Initialize Evaluator for agent performance evaluation
    evaluator = Evaluator(
        high_agent=high_agent,
        low_agent=low_agent,
        X_high_test=X_high_test,
        X_low_test=X_low_test,
        y_test=y_test
    )
    
    # Evaluate the agents on test data
    evaluator.evaluate_agents()

if __name__ == "__main__":
    main()
