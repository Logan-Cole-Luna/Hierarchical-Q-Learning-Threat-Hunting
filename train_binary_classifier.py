# train_binary_classifier.py

import os
import json
import pandas as pd
from agents.binary_agent import BinaryAgent
import logging
from utils.evaluation import evaluate_binary_classifier  # Import evaluation function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Define paths
        binary_train_path = "processed_data/binary_classification/test_binary.csv"
        binary_test_path = "processed_data/binary_classification/train_binary.csv"  # Fixed path
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
        test_df = pd.read_csv(binary_test_path)
        
        # Load label dictionary
        if not os.path.exists(label_dict_path):
            logger.error("Label dictionary not found. Please ensure it exists in the specified path.")
            return
        
        with open(label_dict_path, "r") as infile:
            label_dict = json.load(infile)
        
        # Define feature columns
        feature_cols = [col for col in train_df.columns if col not in ['Threat']]
        
        # Initialize Binary Agent
        binary_agent = BinaryAgent(
            feature_cols=feature_cols,
            label_col='Threat',
            model_path=model_path,
            class_weights_path=class_weights_path,
            label_dict_path=label_dict_path
        )
        
        # Train Binary Classifier
        logger.info("Starting training of Binary Classifier...")
        report = binary_agent.train(train_df, test_df)
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
