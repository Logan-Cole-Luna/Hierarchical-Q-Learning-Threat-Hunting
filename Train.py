# Train.py

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
    # Load mappings
    mappings_dir = os.path.join('data', 'mappings')
    category_to_id_path = os.path.join(mappings_dir, 'category_to_id.json')
    anomaly_to_id_path = os.path.join(mappings_dir, 'anomaly_to_id.json')
    
    if not os.path.exists(category_to_id_path) or not os.path.exists(anomaly_to_id_path):
        print("Error: Mapping files not found. Please run preprocess.py first.")
        sys.exit(1)
    
    # Load preprocessed data
    data_dir = 'data'  # Relative path from main directory
    
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
        high_agent_action_space=None,  # To be set based on mappings
        low_agent_action_space=None,   # To be set based on mappings
        mappings_dir=mappings_dir
    )
    
    # Set action spaces based on mappings
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
    
    # Initialize Trainer
    trainer = Trainer(
        env=env,
        high_agent=high_agent,
        low_agent=low_agent,
        episodes=1000  # Adjust as needed
    )
    
    # Start Training
    trainer.train()
    
    # Save trained models
    high_agent.save('high_level_agent.h5')
    low_agent.save('low_level_agent.h5')
    print("Training completed and models saved.")
    
    # Initialize Evaluator
    evaluator = Evaluator(
        high_agent=high_agent,
        low_agent=low_agent,
        X_high_test=X_high_test,
        X_low_test=X_low_test,
        y_test=y_test
    )
    
    # Evaluate Agents
    evaluator.evaluate_agents()

if __name__ == "__main__":
    main()
