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
from utils.rl_xai_utils import explain_rl_predictions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

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
        # Set device to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Add detailed hardware information
        if device.type == 'cuda':
            num_gpus = torch.cuda.device_count()
            logger.info(f"Number of GPUs available: {num_gpus}")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_properties = torch.cuda.get_device_properties(i)
                total_memory = gpu_properties.total_memory / (1024 ** 3)  # Convert to GB
                logger.info(f"GPU {i}: {gpu_name} with {total_memory:.2f} GB memory")
        else:
            logger.info("No CUDA-compatible GPU found. Using CPU.")

        # Ensure all required directories exist upfront
        os.makedirs("results/multi_class_classification", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        # Define ALL paths at the start
        multi_train_path = "processed_data/multi_class_classification/train_multi_class.csv"
        multi_val_path = "processed_data/multi_class_classification/val_multi_class.csv"  # Load validation data
        multi_test_path = "processed_data/multi_class_classification/test_multi_class.csv"
        label_dict_path = "processed_data/multi_class_classification/label_dict.json"  # Added this line

        # Load preprocessed multi-class classification data
        if not (os.path.exists(multi_train_path) and os.path.exists(multi_test_path)):
            logger.error("Multi-class classification data not found. Please run the preprocessing script first.")
            return
        
        logger.info("Loading multi-class classification data...")
        multi_train_df = pd.read_csv(multi_train_path)
        multi_val_df = pd.read_csv(multi_val_path)    # Load validation data
        multi_test_df = pd.read_csv(multi_test_path)
        logger.info(f"Test set class distribution:\n{multi_test_df['Label'].value_counts()}")
        
        # Drop 'Dst Port' column before training
        multi_train_df = multi_train_df.drop(columns=['Dst Port'])
        multi_val_df = multi_val_df.drop(columns=['Dst Port'])
        multi_test_df = multi_test_df.drop(columns=['Dst Port'])

        # Load label dictionary
        if not os.path.exists(label_dict_path):
            logger.error("Label dictionary not found. Please ensure it exists in the specified path.")
            return
        
        with open(label_dict_path, "r") as infile:
            label_dict = json.load(infile)
        
        # Define feature columns
        multi_feature_cols = [col for col in multi_train_df.columns if col not in ['Label']]
        
        # Save feature columns to JSON for consistent evaluation
        with open('models/rl_dqn_model_features.json', 'w') as f:
            json.dump(multi_feature_cols, f)
        
        # Initialize environment and agent for training
        env = NetworkClassificationEnv(
            multi_train_df, 
            label_dict, 
            batch_size=128  # Set batch_size to match DataLoader's batch_size
        )

        # Initialize environment for validation
        val_env = NetworkClassificationEnv(
            multi_val_df, 
            label_dict, 
            batch_size=128
        )

        state_size = len(multi_feature_cols)  # Number of feature columns
        action_size = len(label_dict)         # Number of unique actions/classes

        # Increase batch size based on GPU memory (e.g., 128)
        agent = Agent(
            state_size=state_size, 
            action_size=action_size, 
            device=device, 
            batch_size=128  # Increased batch size for better GPU utilization
        )  
        agent.qnetwork_local.to(device)
        agent.qnetwork_target.to(device)
        
        print("Q-Network Local Structure:")
        print(agent.qnetwork_local)  # Print network structure
        
        trainer = Trainer(env, agent, val_env=val_env)
        
        num_episodes = 2  # Increased from 100 to 200
        print_interval = get_print_interval(num_episodes)
        logger.info(f"Starting training for {num_episodes} episodes (printing every {print_interval} episodes)...")
        
        reward_history, loss_history = trainer.train(
            num_episodes,
            print_interval=print_interval,
            early_stopping_rounds=1000  # Added early stopping parameter
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
        # Initialize ONNX Session for evaluation
        onnx_model_path = "models/rl_dqn_model.onnx"
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_model_path)

        # Run evaluation with correct parameters
        metrics = evaluate_rl_agent(
            session=session,
            test_df=multi_test_df,
            feature_cols=multi_feature_cols,
            label_col='Label',
            label_mapping=label_dict,
            batch_size=100,
            save_confusion_matrix=True,
            save_roc_curve=True,
            save_path=eval_save_path
        )

        # Set agent back to training mode
        agent.qnetwork_local.train()

        # Log evaluation results
        if metrics:
            logger.info("Evaluation Results:")
            logger.info(f"Accuracy: {metrics.get('accuracy', 'N/A')}")
            
            # Check if per-class metrics exist in the report
            for class_name in metrics:
                if isinstance(metrics[class_name], dict) and 'precision' in metrics[class_name]:
                    logger.info(f"\n{class_name}:")
                    logger.info(f"Precision: {metrics[class_name]['precision']:.4f}")
                    logger.info(f"Recall: {metrics[class_name]['recall']:.4f}")
                    logger.info(f"F1-score: {metrics[class_name]['f1-score']:.4f}")

            # Add XAI analysis after evaluation
            logger.info("\nGenerating RL XAI visualizations...")
            xai_save_path = os.path.join(eval_save_path, 'xai')
            os.makedirs(xai_save_path, exist_ok=True)
            
            # Use RL-specific SHAP explanations
            shap_values = explain_rl_predictions(
                model=session,
                data=multi_test_df[multi_feature_cols].sample(n=1000, random_state=42),  # Sample subset
                feature_names=multi_feature_cols,
                num_samples=50,  # Reduced number of samples
                save_path=xai_save_path,
                max_samples=1000  # Limit maximum samples
            )
            
            if shap_values is not None:
                logger.info("RL XAI visualizations saved to: " + xai_save_path)

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
        
        # After exporting the model to ONNX, ensure it matches the feature dimensions
        onnx_model_path = "models/rl_dqn_model.onnx"
        dummy_input = torch.randn(1, state_size).to(device)
        torch.onnx.export(
            agent.qnetwork_local,
            dummy_input,
            onnx_model_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        logger.info(f"RL model exported to ONNX format at {onnx_model_path}")

    except Exception as e:
        logger.exception("An error occurred during RL agent training.")

if __name__ == "__main__":
    main()
