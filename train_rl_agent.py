# train_rl_agent.py

import os
import json
import torch
import numpy as np
import pandas as pd
from agents.base_agent import Agent
from utils.intrusion_detection_env import NetworkClassificationEnv
from utils.trainer import Trainer
from utils.visualizer import plot_training_metrics
from utils.evaluation import evaluate_rl_agent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Define paths
        multi_train_path = "processed_data/multi_class_classification/train_multi_class.csv"
        multi_test_path = "processed_data/multi_class_classification/test_multi_class.csv"
        label_dict_path = "processed_data/label_dict.json"
        evaluation_save_path = "results/multi_class_classification"

        # Load preprocessed multi-class classification data
        if not (os.path.exists(multi_train_path) and os.path.exists(multi_test_path)):
            logger.error("Multi-class classification data not found. Please run the preprocessing script first.")
            return
        
        logger.info("Loading multi-class classification data...")
        multi_train_df = pd.read_csv(multi_train_path)
        multi_test_df = pd.read_csv(multi_test_path)
        
        # Load label dictionary
        if not os.path.exists(label_dict_path):
            logger.error("Label dictionary not found. Please ensure it exists in the specified path.")
            return
        
        with open(label_dict_path, "r") as infile:
            label_dict = json.load(infile)
        
        # Define feature columns
        multi_feature_cols = [col for col in multi_train_df.columns if col not in ['Label']]
        
        # Initialize Environment for RL Agent
        env_batch_size = 64
        env = NetworkClassificationEnv(multi_train_df, label_dict, batch_size=env_batch_size)
        
        # Initialize RL Agent
        rl_agent = Agent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            hidden_layers=[128, 64],
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            batch_size=64,
            memory_size=10000,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Initialize Trainer for RL Agent
        rl_trainer = Trainer(env, rl_agent)
        
        num_episodes = 300
        logger.info(f"Starting RL training for {num_episodes} episodes...")
        
        start_time = time.time()
        rl_reward_history, rl_loss_history = rl_trainer.train(num_episodes)
        end_time = time.time()
        
        logger.info(f"RL Training completed in {(end_time - start_time)/60:.2f} minutes.")
        
        # Plot RL Training Metrics
        rl_metrics_plot_path = os.path.join("results", "multi_class_classification", "rl_training_metrics.png")
        os.makedirs(os.path.dirname(rl_metrics_plot_path), exist_ok=True)
        plot_training_metrics(rl_reward_history, rl_loss_history, rl_metrics_plot_path)
        logger.info(f"RL Training metrics saved to {rl_metrics_plot_path}")
        
        # Save RL Model
        os.makedirs("models", exist_ok=True)
        rl_model_path = "models/rl_dqn_model.pth"
        torch.save(rl_agent.qnetwork_local.state_dict(), rl_model_path)
        logger.info(f"RL Model saved to {rl_model_path}")
        
        # Evaluate RL Agent
        eval_env = NetworkClassificationEnv(multi_test_df, label_dict, batch_size=64)
        rl_metrics = evaluate_rl_agent(
            agent=rl_agent,
            env=eval_env,
            label_dict=label_dict,
            multi_test_df=multi_test_df,
            save_confusion_matrix=True,
            save_roc_curves=True,
            save_path='results/multi_class_classification'
        )
        logger.info("RL Evaluation completed.")
        
    except Exception as e:
        logger.exception
