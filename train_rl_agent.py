"""
Main training script for the reinforcement learning agent.

This script orchestrates the complete training pipeline including:
1. Data loading and preparation
2. Environment and agent initialization
3. Training loop management
4. Model evaluation
5. Results saving and visualization

The script supports both binary and multi-class classification tasks and
includes comprehensive logging and error handling.

Key Features:
- Configurable training parameters
- Progress tracking and logging
- Model checkpointing
- Performance evaluation
- Results visualization
"""

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
import time
from collections import Counter
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data():
    # Load data
    df = pd.read_csv('processed_data/multi_class_classification/train_multi_class.csv')
    
    # Ensure balanced split
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # Stratify by Label to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Debug prints
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    logger.info(f"Training labels distribution:\n{y_train.value_counts()}")
    logger.info(f"Test labels distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test

def get_print_interval(num_episodes):
    """Determine appropriate printing interval based on number of episodes."""
    if num_episodes <= 10:
        return 1  # Print every episode
    elif num_episodes <= 50:
        return 5  # Print every 5th episode
    elif num_episodes <= 100:
        return 10  # Print every 10th episode
    else:
        return num_episodes // 10  # Print 10 times total

def main():
    try:
        # Ensure all required directories exist upfront
        os.makedirs("results/multi_class_classification", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        # Define ALL paths at the start
        multi_train_path = "processed_data/multi_class_classification/train_multi_class_subset.csv"
        multi_test_path = "processed_data/multi_class_classification/train_multi_class_subset.csv"
        label_dict_path = "processed_data/multi_class_classification/label_dict.json"  # Added this line
        evaluation_save_path = os.path.abspath('results/multi_class_classification')

        # Load preprocessed multi-class classification data
        if not (os.path.exists(multi_train_path) and os.path.exists(multi_test_path)):
            logger.error("Multi-class classification data not found. Please run the preprocessing script first.")
            return
        
        logger.info("Loading multi-class classification data...")
        multi_train_df = pd.read_csv(multi_train_path)
        multi_test_df = pd.read_csv(multi_test_path)
        logger.info(f"Test set class distribution:\n{multi_test_df['Label'].value_counts()}")
        
        # Load label dictionary
        if not os.path.exists(label_dict_path):
            logger.error("Label dictionary not found. Please ensure it exists in the specified path.")
            return
        
        with open(label_dict_path, "r") as infile:
            label_dict = json.load(infile)
        
        # Define feature columns
        multi_feature_cols = [col for col in multi_train_df.columns if col not in ['Label']]
        
        # Initialize environment and agent for training
        env = NetworkClassificationEnv(multi_train_df, label_dict)
        state_size = len(multi_feature_cols)  # Number of feature columns
        action_size = len(label_dict)         # Number of unique actions/classes

        agent = Agent(state_size=state_size, action_size=action_size)
        trainer = Trainer(env, agent)
    
        # Start training
        num_episodes = 150
        print_interval = get_print_interval(num_episodes)
        logger.info(f"Starting training for {num_episodes} episodes (printing every {print_interval} episodes)...")
        
        reward_history, loss_history = trainer.train(num_episodes, print_interval=print_interval)
    
        # Ensure all classes are included in the dataset
        class_counts = Counter(multi_train_df['Label'])
        print(f"Class distribution: {class_counts}")

        # Modify loss function for multi-class
        criterion = nn.CrossEntropyLoss()

        # Print final results
        logger.info("\nTraining completed.")
        logger.info(f"Final reward: {reward_history[-1]:.2f}")
        logger.info(f"Final loss: {loss_history[-1]:.4f}")

        # Initialize Environment for RL Agent evaluation
        eval_env = NetworkClassificationEnv(
            multi_test_df, 
            label_dict, 
            batch_size=100,  # Large batch for faster evaluation
            max_steps=len(multi_test_df)  # Allow evaluation on full test set
        )

        # Set agent to evaluation mode
        agent.qnetwork_local.eval()
        
        # Create evaluation directory
        eval_save_path = os.path.abspath('results/multi_class_classification')
        os.makedirs(eval_save_path, exist_ok=True)
        logger.info(f"Saving evaluation results to: {eval_save_path}")
        
        # Run evaluation
        metrics = evaluate_rl_agent(
            agent=agent,
            env=eval_env,
            label_dict=label_dict,
            multi_test_df=multi_test_df,
            batch_size=100,
            save_confusion_matrix=True,
            save_roc_curves=True,
            save_path=eval_save_path
        )

        # Set agent back to training mode
        agent.qnetwork_local.train()

        # Log evaluation results
        if metrics:
            logger.info("Evaluation Results:")
            logger.info(f"Accuracy: {metrics['classification_report']['accuracy']:.4f}")
            logger.info("\nPer-class Results:")
            for class_name in label_dict.keys():
                class_metrics = metrics['classification_report'][class_name]
                logger.info(f"\n{class_name}:")
                logger.info(f"Precision: {class_metrics['precision']:.4f}")
                logger.info(f"Recall: {class_metrics['recall']:.4f}")
                logger.info(f"F1-score: {class_metrics['f1-score']:.4f}")

        # Plot RL Training Metrics
        rl_metrics_plot_path = os.path.join("results", "multi_class_classification", "rl_training_metrics.png")
        os.makedirs(os.path.dirname(rl_metrics_plot_path), exist_ok=True)
        plot_training_metrics(reward_history, loss_history, rl_metrics_plot_path)
        logger.info(f"RL Training metrics saved to {rl_metrics_plot_path}")
        
        # Save RL Model
        os.makedirs("models", exist_ok=True)
        rl_model_path = "models/rl_dqn_model.pth"
        torch.save(agent.qnetwork_local.state_dict(), rl_model_path)
        logger.info(f"RL Model saved to {rl_model_path}")
        
    except Exception as e:
        logger.exception("An error occurred during RL agent training.")

if __name__ == "__main__":
    main()
