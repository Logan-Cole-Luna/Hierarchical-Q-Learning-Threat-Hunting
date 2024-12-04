"""
Training script for the binary classifier in the hierarchical classification system.

This script handles the training of a binary classifier that distinguishes between
benign and malicious network traffic. It serves as the first stage in the
hierarchical classification system.

Key Features:
- Loads preprocessed binary classification data
- Initializes and trains binary classifier
- Evaluates model performance
- Saves model and evaluation results
"""

# train_binary_classifier.py

import os
import json
import pandas as pd
from agents.binary_agent import BinaryAgent
import logging
from utils.evaluation import evaluate_binary_classifier  # Import evaluation function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Define paths
        binary_train_path = "processed_data/binary_classification/train_binary.csv"
        binary_val_path = "processed_data/binary_classification/val_binary.csv"    # Load validation data
        binary_test_path = "processed_data/binary_classification/test_binary.csv"
        label_dict_path = "processed_data/binary_classification/label_dict.json"
        class_weights_path = "processed_data/binary_classification/class_weights.json"
        model_path = "models/binary_classifier.pkl"
        evaluation_save_path = "results/binary_classification"

        # Load preprocessed binary classification data
        if not (os.path.exists(binary_train_path) and os.path.exists(binary_test_path)):
            logger.error("Binary classification data not found. Please run the preprocessing script first.")
            return
        
        logger.info("Loading binary classification data...")
        train_df = pd.read_csv(binary_train_path)
        val_df = pd.read_csv(binary_val_path)    # Load validation data
        test_df = pd.read_csv(binary_test_path)
        
        # Ensure no overlap between train and test sets
        # Replace overlap check with hash-based method
        # overlap_exists = train_hashes.isin(test_hashes).any()
        # if overlap_exists:
        #     overlapping_records = train_df[train_hashes.isin(test_hashes)]
        #     logger.error("Overlap detected between training and testing sets.")
        #     logger.debug(f"Number of overlapping records: {overlapping_records.shape[0]}")
        #     logger.debug(f"Overlapping records:\n{overlapping_records.head()}")
        #     return
        
        # Load label dictionary
        # Removed loading label_dict since it's handled in BinaryAgent
        # if not os.path.exists(label_dict_path):
        #     logger.error("Label dictionary not found. Please ensure it exists in the specified path.")
        #     return
        
        # with open(label_dict_path, "r") as infile:
        #     label_dict = json.load(infile)
        
        # Define feature columns
        feature_cols = [col for col in train_df.columns if col not in ['Threat']]
        
        # Initialize Binary Agent
        binary_agent = BinaryAgent(
            feature_cols=feature_cols,
            label_col='Threat',
            model_path=model_path,
            class_weights_path=class_weights_path,
            label_dict_path=label_dict_path  # This parameter can remain or be removed if not used elsewhere
        )
        
        # Train Binary Classifier with validation data
        logger.info("Starting training of Binary Classifier...")
        report = binary_agent.train(train_df, test_df, val_df=val_df)  # Corrected argument order
        logger.info("Binary Classifier Training and Evaluation Completed.")
        
        # Ensure probabilities are obtained for ROC computation
        X_test = test_df[feature_cols]
        y_scores = binary_agent.model.predict_proba(X_test)
        
        # Evaluate Binary Classifier
        logger.info("Evaluating Binary Classifier on Test Data...")
        binary_report = report  # Already handled within BinaryAgent
        logger.info("Binary Classification Evaluation Metrics:")
        if isinstance(binary_report, dict):
            for key, value in binary_report.items():
                logger.info(f"{key}: {value}")
        else:
            logger.error("binary_report is not a dictionary.")
        
        # Call evaluation function to save performance metrics and visuals
        evaluate_binary_classifier(binary_agent, test_df, save_path='results/binary_classification')
    except Exception as e:
        logger.exception("An error occurred during binary classifier training.")

if __name__ == "__main__":
    main()
