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

def evaluate_model(y_true, y_pred):
    # Verify labels before evaluation
    logger.info(f"Label value ranges - y_true: {np.unique(y_true)}, y_pred: {np.unique(y_pred)}")
    
    # Load label mapping
    with open('processed_data/multi_class_classification/label_dict.json', 'r') as f:
        label_dict = json.load(f)
    
    # Convert numeric labels to string labels for better readability
    inv_label_dict = {v: k for k, v in label_dict.items()}
    y_true_labels = [inv_label_dict[y] if y in inv_label_dict else None for y in y_true]
    y_pred_labels = [inv_label_dict[y] if y in inv_label_dict else None for y in y_pred]
    
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
        reward_history, loss_history = trainer.train(num_episodes)
    
        # Ensure all classes are included in the dataset
        class_counts = Counter(multi_train_df['Label'])
        print(f"Class distribution: {class_counts}")

        # Modify loss function for multi-class
        criterion = nn.CrossEntropyLoss()

        # Initialize Environment for RL Agent evaluation (batch_size=1)
        eval_env = NetworkClassificationEnv(multi_test_df, label_dict, batch_size=100)  # Increased batch size
        
        # Initialize Trainer for evaluation
        trainer_eval = Trainer(eval_env, agent)
        
        # Evaluation using evaluate_rl_agent with absolute path
        eval_save_path = os.path.abspath('results/multi_class_classification')
        logger.info(f"Saving evaluation results to: {eval_save_path}")
        
        metrics = evaluate_rl_agent(
            agent=agent,
            env=eval_env,
            label_dict=label_dict,
            multi_test_df=multi_test_df,
            batch_size=100,  # Increased batch size
            save_confusion_matrix=True,
            save_roc_curves=True,
            save_path=eval_save_path  # Use absolute path
        )

        # Log evaluation results
        logger.info("Evaluation completed. Results:")
        for metric_name, metric_value in metrics['classification_report'].items():
            if isinstance(metric_value, dict):
                logger.info(f"\n{metric_name}:")
                for k, v in metric_value.items():
                    logger.info(f"{k}: {v}")
            else:
                logger.info(f"{metric_name}: {metric_value}")
        
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
