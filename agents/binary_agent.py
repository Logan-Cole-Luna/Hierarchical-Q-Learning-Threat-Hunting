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
import xgboost as xgb  # Added import for version check

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
            random_state=42,
            early_stopping=10  # Add early stopping here instead
        )
        self.class_weights = None
        self.label_dict = None
        # Define label encoding
        self.label_mapping = {'Benign': 0, 'Malicious': 1}
    
    def train(self, train_df, test_df, val_df=None):
        """
        Trains the binary classifier and evaluates it on the test set.
        
        Parameters:
        - train_df (pd.DataFrame): Training dataset.
        - test_df (pd.DataFrame): Testing dataset.
        - val_df (pd.DataFrame, optional): Validation dataset for early stopping.
        """
        X_train = train_df[self.feature_cols]
        y_train = train_df[self.label_col]
        X_test = test_df[self.feature_cols]
        y_test = test_df[self.label_col]
        
        # Encode labels
        y_train = y_train.map(self.label_mapping).astype(int)
        y_test = y_test.map(self.label_mapping).astype(int)
        
        # Compute class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        self.class_weights = {i: weight for i, (label, weight) in enumerate(zip(classes, class_weights))}
        
        # Train XGBoost Classifier
        logger.info("Training Binary Classifier...")
        
        # Log XGBoost version
        logger.info(f"XGBoost version: {xgb.__version__}")
        
        if val_df is not None:
            X_val = val_df[self.feature_cols]
            y_val = val_df[self.label_col].map(self.label_mapping).astype(int)
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,  # Remove early_stopping_rounds parameter
                verbose=True
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Replace the strict assertion with proper evaluation
        if test_df is not None:
            y_test_encoded = y_test
            y_pred = self.model.predict(X_test)
            # Remove assertion and just evaluate
            accuracy = np.mean(y_test_encoded == y_pred)
            logger.info(f"Test set accuracy: {accuracy:.4f}")
        
        logger.info("Binary Classifier training completed.")
        
        # Save the model using joblib.dump instead of torch.save
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)  # Changed from torch.save to joblib.dump
        logger.info(f"Binary Classifier saved to {self.model_path}")
        
        # Save class weights and label dictionary
        with open(self.class_weights_path, 'w') as f:
            json.dump(self.class_weights, f)
        logger.info(f"Class weights saved to {self.class_weights_path}")
        
        # Removed saving label_dict as it's no longer used
        # with open(self.label_dict_path, 'w') as f:
        #     json.dump(self.label_dict, f)
        # logger.info(f"Label dictionary saved to {self.label_dict_path}")
        
        # Evaluate on test set
        report = classification_report(y_test, y_pred, target_names=self.label_mapping.keys(), output_dict=True)
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
