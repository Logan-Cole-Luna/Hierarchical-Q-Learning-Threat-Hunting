# agents/binary_agent.py

import os
import json
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import logging
import numpy as np
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import torch

logger = logging.getLogger(__name__)

class BinaryAgent:
    def __init__(self, feature_cols, label_col, model_path, class_weights_path, label_dict_path):
        """
        Initializes the BinaryAgent with feature columns and label information.
        
        Parameters:
        - feature_cols (list): List of feature column names.
        - label_col (str): Name of the binary label column ('Threat').
        - model_path (str): Path to save the trained binary classifier.
        - class_weights_path (str): Path to save class weights.
        - label_dict_path (str): Path to save the label dictionary.
        """
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.model_path = model_path
        self.class_weights_path = class_weights_path
        self.label_dict_path = label_dict_path
        self.model = XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        self.class_weights = None
        self.label_dict = None
    
    def train(self, train_df, test_df):
        """
        Trains the binary classifier and evaluates it on the test set.
        
        Parameters:
        - train_df (pd.DataFrame): Training dataset.
        - test_df (pd.DataFrame): Testing dataset.
        """
        X_train = train_df[self.feature_cols]
        y_train = train_df[self.label_col]
        X_test = test_df[self.feature_cols]
        y_test = test_df[self.label_col]
        
        # Compute class weights
        classes = np.unique(y_train)
        self.label_dict = {label: idx for idx, label in enumerate(classes)}
        y_train_encoded = y_train.map(self.label_dict).astype(int)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        self.class_weights = {i: weight for i, (label, weight) in enumerate(zip(classes, class_weights))}
        
        # Train XGBoost Classifier
        logger.info("Training Binary Classifier...")
        self.model.fit(
            X_train, y_train_encoded,
            sample_weight=[self.class_weights[idx] for idx in y_train_encoded]
        )
        logger.info("Binary Classifier training completed.")
        
        # Save the model using joblib.dump instead of torch.save
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)  # Changed from torch.save to joblib.dump
        logger.info(f"Binary Classifier saved to {self.model_path}")
        
        # Save class weights and label dictionary
        with open(self.class_weights_path, 'w') as f:
            json.dump(self.class_weights, f)
        logger.info(f"Class weights saved to {self.class_weights_path}")
        
        with open(self.label_dict_path, 'w') as f:
            json.dump(self.label_dict, f)
        logger.info(f"Label dictionary saved to {self.label_dict_path}")
        
        # Evaluate on test set
        y_test_encoded = y_test.map(self.label_dict).astype(int)
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test_encoded, y_pred, target_names=classes, output_dict=True)
        logger.info("Binary Classifier Evaluation Report:")
        logger.info("\n" + json.dumps(report, indent=2))
        return report
    
    def export_to_onnx(self, onnx_model_path, feature_cols):
        """
        Exports the trained XGBoost model to ONNX format.
        
        Parameters:
        - onnx_model_path (str): Path to save the ONNX model.
        - feature_cols (list): List of feature column names.
        """
        logger.info("Exporting Binary Classifier to ONNX format...")
        initial_type = [('float_input', FloatTensorType([None, len(feature_cols)]))]
        onnx_model = convert_sklearn(self.model, initial_types=initial_type)
        with open(onnx_model_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info(f"Binary Classifier exported to ONNX at {onnx_model_path}")
    
    def predict(self, X):
        """
        Predicts binary labels for given input features.
        
        Parameters:
        - X (pd.DataFrame): Input features.
        
        Returns:
        - predictions (np.ndarray): Predicted binary labels.
        """
        return self.model.predict(X)
    
    def predict_batch(self, X_batch):
        """
        Predicts labels and returns both predictions and prediction probabilities for a batch of data.
        
        Parameters:
        - X_batch (np.ndarray): Batch of input features.
    
        Returns:
        - preds (np.ndarray): Predicted labels.
        - scores (np.ndarray): Prediction probabilities.
        """
        preds = self.model.predict(X_batch)
        scores = self.model.predict_proba(X_batch)  # Ensure probabilities are returned
        return preds, scores
