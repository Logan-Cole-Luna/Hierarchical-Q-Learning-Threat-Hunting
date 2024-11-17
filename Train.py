# Train.py

import os
import json
import time
import torch
import numpy as np
import pandas as pd
from agents.base_agent import Agent
from utils.intrusion_detection_env import NetworkClassificationEnv
from utils.trainer import Trainer
from utils.visualizer import plot_training_metrics
from utils.evaluation import evaluate_agent
import logging
from scripts.preprocess import preprocess_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_preprocessed_data():
    """
    Loads preprocessed training and testing data along with the label dictionary.
    If the data does not exist, it returns None.
    """
    train_path = "data/train_df.csv"
    test_path = "data/test_df.csv"
    label_dict_path = "data/mappings/label_dict.json"
    
    if os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(label_dict_path):
        logger.info("Loading preprocessed data...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        with open(label_dict_path, "r") as infile:
            label_dict = json.load(infile)
        logger.info("Preprocessed data loaded successfully.")
        return train_df, test_df, label_dict
    else:
        logger.warning("Preprocessed data not found.")
        return None

def main():
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(0)
        np.random.seed(0)
        
        # Attempt to load preprocessed data
        data = load_preprocessed_data()
        
        if data is None:
            # If preprocessed data does not exist, run preprocess.py
            logger.info("Preprocessing data as preprocessed files were not found...")
            train_df, test_df, feature_cols, label_dict = preprocess_data()
        else:
            train_df, test_df, label_dict = data
            # Extract feature columns excluding label columns
            feature_cols = [col for col in train_df.columns if col not in ['Label', 'Threat']]
        
        # Initialize environment and agent
        logger.info("Initializing environment and agent...")
        env_batch_size = 64  # Define env_batch_size before using it
        env = NetworkClassificationEnv(train_df, label_dict, batch_size=env_batch_size)
        agent = Agent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            hidden_layers=[64, 32],
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            batch_size=64,
            memory_size=10000,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        trainer = Trainer(env, agent)
        
        num_episodes = 300
        logger.info(f"Starting training for {num_episodes} episodes...")
        
        start_time = time.time()
        reward_history, loss_history = trainer.train(num_episodes)
        end_time = time.time()
        
        logger.info(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")
        
        # Plot training metrics
        plot_training_metrics(reward_history, loss_history, os.path.join("results", "training_metrics.png"))
        logger.info("Training metrics saved to results/training_metrics.png")
        
        # Save the trained model
        os.makedirs("models", exist_ok=True)
        model_path = "models/dqn_model.pth"
        torch.save(agent.qnetwork_local.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Evaluate the agent
        # Initialize a new environment for evaluation if needed
        eval_env = NetworkClassificationEnv(test_df, label_dict, batch_size=64)
        metrics = evaluate_agent(
            agent=agent,
            env=eval_env,
            label_dict=label_dict,
            test_df=test_df,
            save_confusion_matrix=True,
            save_roc_curves=True,
            save_path='results'
        )
        logger.info("Evaluation completed.")
    
    except Exception as e:
        logger.exception("An error occurred during training.")

if __name__ == "__main__":
    main()
