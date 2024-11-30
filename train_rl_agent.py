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
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data():
    # Load the preprocessed training and testing data
    train_data = pd.read_csv('processed_data/multi_class_classification/train_multi_class.csv')
    test_data = pd.read_csv('processed_data/multi_class_classification/test_multi_class.csv')

    # Extract features and labels
    feature_cols = [col for col in train_data.columns if col not in ['Label', 'Threat', 'Timestamp']]
    X_train = train_data[feature_cols].values
    y_train = train_data['Label'].values
    X_test = test_data[feature_cols].values
    y_test = test_data['Label'].values

    # Remove the reshaping code since the data is not in sequence format
    # sequence_length = 10  # Set the sequence length used during preprocessing
    # num_features = X_train.shape[1] // sequence_length
    # X_train = X_train.reshape(-1, sequence_length, num_features)
    # X_test = X_test.reshape(-1, sequence_length, num_features)

    # After loading data
    print(f"[DEBUG] Loaded data shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

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
        multi_train_path = "processed_data/multi_class_classification/train_multi_class.csv"
        multi_test_path = "processed_data/multi_class_classification/test_multi_class.csv"  # Changed from train to test
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
        
        # Create inverse label dictionary
        inv_label_dict = {v: k for k, v in label_dict.items()}
        
        # Define feature columns
        multi_feature_cols = [col for col in multi_train_df.columns if col not in ['Label']]
        
        # Initialize environment and agent for training
        env = NetworkClassificationEnv(multi_train_df, label_dict)
        state_size = 44  # Updated to match the number of features
        action_size = len(label_dict)         # Number of unique actions/classes

        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_and_preprocess_data()

        # *** Begin Changes ***
        # Map label strings to integers using label_dict
        y_train = [label_dict.get(label, -1) for label in y_train]
        y_test = [label_dict.get(label, -1) for label in y_test]
        
        # Check for any unmapped labels
        if -1 in y_train or -1 in y_test:
            logger.error("Found labels that are not in label_dict. Please check your label mappings.")
            return

        # Convert labels to numpy arrays
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        # Print shapes and unique labels after mapping
        print(f"[DEBUG] After mapping labels:")
        print(f"y_train.shape = {y_train.shape}")
        print(f"y_test.shape = {y_test.shape}")
        print(f"Unique y_train labels: {np.unique(y_train)}")
        print(f"Unique y_test labels: {np.unique(y_test)}")

        # After mapping labels
        logger.debug(f"Label dict: {label_dict}")
        logger.debug(f"Unique y_train labels: {np.unique(y_train)}")
        logger.debug(f"Unique y_test labels: {np.unique(y_test)}")

        # Assertions to ensure labels are correctly encoded
        assert all(label in label_dict.values() for label in y_train), "Some training labels are missing from label_dict."
        assert all(label in label_dict.values() for label in y_test), "Some testing labels are missing from label_dict."
        # *** End Changes ***

        # Determine the actual number of features after preprocessing
        print(f"[DEBUG] Before accessing X_train.shape[2]: X_train.shape = {X_train.shape}")
        state_size = X_train.shape[1]  # Changed from X_train.shape[2] to X_train.shape[1]
        print(f"State size (number of input features): {state_size}")

        # Initialize agent and trainer
        agent = Agent(state_size=state_size, action_size=action_size)
        trainer = Trainer(env, agent)
    
        # Start training
        num_episodes = 5
        print_interval = get_print_interval(num_episodes)
        logger.info(f"Starting training for {num_episodes} episodes (printing every {print_interval} episodes)...")
        
        reward_history, loss_history = trainer.train(
            num_episodes=num_episodes,
            print_interval=print_interval,
            X_test=X_test,
            y_test=y_test
        )
    
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
        logger.info("Starting evaluation of RL Agent...")
        metrics = evaluate_rl_agent(
            y_true=y_test,  # y_test should be integer-encoded
            y_pred=agent.predict(X_test),  # Ensure predictions are also integer-encoded
            label_dict_path=label_dict_path  # Provide the path to label_dict.json
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
        logger.error(f"An error occurred during RL agent training.\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
