"""
Evaluation script for both binary and multi-class models.

This script loads trained models and evaluates their performance on test data,
generating comprehensive evaluation metrics and visualizations.
"""

import os
import json
import logging
import torch
import pandas as pd
from agents.binary_agent import BinaryAgent
from agents.base_agent import Agent
from utils.evaluation import evaluate_binary_classifier, evaluate_rl_agent
import onnxruntime as ort  # Import ONNX Runtime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_onnx_model(model_path):
    """Loads an ONNX model and creates an inference session."""
    if not os.path.exists(model_path):
        logger.error(f"ONNX model file '{model_path}' does not exist.")
        return None
    session = ort.InferenceSession(model_path)
    return session

def evaluate_binary_classifier_onnx(session, test_df, feature_cols, label_col, label_mapping):
    """Evaluates the binary classifier using the ONNX model."""
    import numpy as np

    X = test_df[feature_cols].values.astype(np.float32)
    inputs = {session.get_inputs()[0].name: X}
    outputs = session.run(None, inputs)
    y_pred = np.argmax(outputs[0], axis=1)

    # Map predictions back to original labels
    inverse_label_dict = {v: k for k, v in label_mapping.items()}
    y_pred_labels = [inverse_label_dict[idx] for idx in y_pred]

    # Generate classification report
    from sklearn.metrics import classification_report
    report = classification_report(test_df[label_col], y_pred_labels, output_dict=True)
    return report

def evaluate_rl_agent_onnx(session, test_df, feature_cols, label_col, label_mapping):
    """Evaluates the RL agent using the ONNX model."""
    import numpy as np

    X = test_df[feature_cols].values.astype(np.float32)
    inputs = {session.get_inputs()[0].name: X}
    outputs = session.run(None, inputs)
    y_pred = np.argmax(outputs[0], axis=1)

    # Map predictions back to original labels
    inverse_label_dict = {v: k for k, v in label_mapping.items()}
    y_pred_labels = [inverse_label_dict[idx] for idx in y_pred]

    # Generate classification report
    from sklearn.metrics import classification_report
    report = classification_report(test_df[label_col], y_pred_labels, output_dict=True)
    return report

def main():
    # Paths to test data
    binary_test_path = 'processed_data/binary_classification/test_binary.csv'
    multi_test_path = 'processed_data/multi_class_classification/test_multi_class.csv'

    # Paths to models
    binary_model_path = 'models/binary_classifier.pkl'
    rl_model_path = 'models/rl_dqn_model.pth'

    # Load test data
    binary_test_df = pd.read_csv(binary_test_path)
    multi_test_df = pd.read_csv(multi_test_path)

    # Define feature columns based on the dataset
    binary_feature_cols = [col for col in binary_test_df.columns if col not in ['Label', 'Threat']]

    # Load label dictionaries
    with open('processed_data/binary_classification/label_dict.json', 'r') as f:
        binary_label_dict = json.load(f)
    with open('processed_data/multi_class_classification/label_dict.json', 'r') as f:
        multi_label_dict = json.load(f)

    # Initialize binary agent
    binary_agent = BinaryAgent(
        feature_cols=binary_feature_cols,
        label_col='Threat',
        model_path=binary_model_path,
        class_weights_path='models/binary_class_weights.json',
        label_dict_path='models/binary_label_dict.json'
    )

    # Load feature columns from JSON to ensure consistency
    with open('models/rl_dqn_model_features.json', 'r') as f:
        multi_feature_cols = json.load(f)

    # Initialize RL agent for multi-class classification
    rl_agent = Agent(
        state_size=len(multi_feature_cols),
        action_size=len(multi_label_dict),
        hidden_layers=[128, 64],
        device=torch.device('cpu')
    )
    # Load the RL agent's model
    rl_agent.qnetwork_local.load_state_dict(torch.load(rl_model_path, map_location='cpu'))
    rl_agent.qnetwork_local.eval()

    # Load ONNX models
    binary_onnx_path = 'models/binary_classifier.onnx'
    rl_onnx_path = 'models/rl_dqn_model.onnx'
    '''
    binary_session = None #load_onnx_model(binary_onnx_path)
    if binary_session is None:
        logger.error("Binary Classifier ONNX model could not be loaded. Please ensure the model is exported correctly.")
        return
    '''
    rl_session = load_onnx_model(rl_onnx_path)
    logger.info(f"Loaded RL Model ONNX model from {rl_onnx_path}")
    '''
    # Evaluate binary classifier using ONNX
    logger.info("Evaluating Binary Classifier using ONNX model...")
    binary_report = evaluate_binary_classifier_onnx(
        session=binary_session,
        test_df=binary_test_df,
        feature_cols=binary_feature_cols,
        label_col='Threat',
        label_mapping=binary_label_dict
    )
    '''
    
    # Evaluate RL agent using ONNX
    logger.info("Evaluating Multi-Class Classifier using ONNX model...")
    multi_report = evaluate_rl_agent_onnx(
        session=rl_session,
        test_df=multi_test_df,
        feature_cols=multi_feature_cols,
        label_col='Label',
        label_mapping=multi_label_dict
    )

    # Log evaluation reports
    logger.info("Binary Classification Report:")
    #logger.info("\n" + json.dumps(binary_report, indent=2))
    logger.info("Multi-Class Classification Report:")
    logger.info("\n" + json.dumps(multi_report, indent=2))

if __name__ == "__main__":
    main()