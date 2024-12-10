"""
Reinforcement Learning Explainable AI (XAI) Utilities

This module provides utilities for generating explanations of RL model predictions
using SHAP (SHapley Additive exPlanations) values. It extends the binary classification
implementation in binary_xai_utils.py to handle multi-class RL scenarios.

Key Features:
- SHAP value generation for RL models via KernelExplainer
- Multi-class misclassification analysis
- Feature importance visualization across all classes
- Human-readable explanations of RL agent decisions

See Also:
    - utils/binary_xai_utils.py: Similar implementation for binary classification
    - utils/evaluation.py: Uses these utilities for model evaluation
"""

import shap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
import torch
import onnxruntime as ort
from typing import List, Dict
import pandas as pd

# Ensure matplotlib is using a suitable backend
matplotlib.use('TkAgg')  # Or 'Agg' if running without a display server

logger = logging.getLogger(__name__)

def explain_rl_predictions(model, data: pd.DataFrame, feature_names: List[str], 
                         num_samples: int = 100, save_path: str = None,
                         max_samples: int = 1000):
    """
    Generate SHAP explanations for RL model predictions.
    
    Similar to explain_binary_predictions() but uses KernelExplainer for ONNX models
    and handles multi-class predictions. See binary_xai_utils.py for binary version.

    Args:
        model: ONNX model for RL agent
        data: Feature data to explain
        feature_names: Names of input features
        num_samples: Number of background samples for SHAP
        save_path: Directory to save visualizations
        max_samples: Maximum samples to analyze (prevents memory issues)

    Returns:
        numpy.ndarray: SHAP values if successful, None if failed
    """
    try:
        # Subsample data if needed
        if len(data) > max_samples:
            data = data.sample(n=max_samples, random_state=42)
            logger.info(f"Subsampled data to {max_samples} samples for SHAP analysis")
        
        if isinstance(model, ort.InferenceSession):
            # Configure ONNX prediction wrapper
            data_values = data.values.astype(np.float32)
            logger.debug(f"Data values shape: {data_values.shape}")
            
            # Create smaller background dataset and get underlying data
            background_data = shap.kmeans(data_values[:min(30, len(data))], k=3)
            background_values = background_data.data
            logger.debug(f"Background values shape: {background_values.shape}")
            
            def model_predict(x):
                try:
                    if isinstance(x, shap.utils._legacy.DenseData):
                        x = x.data
                    outputs = model.run(None, {model.get_inputs()[0].name: x.astype(np.float32)})[0]
                    exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
                    probs = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
                    logger.debug(f"Model prediction output shape: {probs.shape}")
                    return probs
                except Exception as e:
                    logger.error(f"Error in model prediction: {str(e)}")
                    return None
            
            # Initialize and test KernelExplainer
            test_pred = model_predict(background_values)
            if test_pred is None:
                logger.error("Model prediction failed on background data")
                return None
            logger.debug(f"Test prediction shape: {test_pred.shape}")
            
            explainer = shap.KernelExplainer(
                model_predict,
                background_values,  # Use background_values instead of background
                link="identity",
                feature_perturbation="interventional"
            )
            logger.info("KernelExplainer created successfully")
            
            # Generate and validate SHAP values
            try:
                sample_data = data_values[:min(10, len(data))]
                shap_values = explainer.shap_values(
                    sample_data,
                    nsamples=50
                )
                logger.info(f"SHAP values type: {type(shap_values)}")
                if isinstance(shap_values, list):
                    logger.info(f"Number of SHAP value arrays: {len(shap_values)}")
                    for i, sv in enumerate(shap_values):
                        logger.info(f"SHAP values[{i}] shape: {sv.shape}")
                else:
                    logger.info(f"SHAP values shape: {shap_values.shape}")
            except Exception as e:
                logger.error(f"Error generating SHAP values: {str(e)}")
                return None
            
            # Generate visualizations if requested
            if save_path and shap_values is not None:
                try:
                    # Handle 3D SHAP values for multi-class
                    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                        # Average absolute SHAP values across classes
                        avg_shap = np.abs(shap_values).mean(axis=2)
                        
                        # Bar plot
                        plt.figure(figsize=(12, 8))
                        shap.summary_plot(
                            avg_shap,
                            data_values[:min(10, len(data))],
                            feature_names=feature_names,
                            plot_type="bar",
                            show=False
                        )
                        plt.title("RL Model SHAP Summary (Impact Magnitude)")
                        plt.tight_layout()
                        summary_path = f"{save_path}/rl_shap_summary.png"
                        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        # Beeswarm plot
                        plt.figure(figsize=(12, 8))
                        shap.summary_plot(
                            avg_shap,
                            data_values[:min(10, len(data))],
                            feature_names=feature_names,
                            show=False
                        )
                        plt.title("RL Model SHAP Summary (Feature Values)")
                        plt.tight_layout()
                        beeswarm_path = f"{save_path}/rl_shap_beeswarm.png"
                        plt.savefig(beeswarm_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        logger.info(f"RL SHAP summary plots saved to {save_path}")
                    else:
                        logger.warning(f"Unexpected SHAP values format: {type(shap_values)}")
                        if isinstance(shap_values, np.ndarray):
                            logger.warning(f"Shape: {shap_values.shape}")
                    
                except Exception as e:
                    logger.error(f"Error creating SHAP plot: {str(e)}")
            
            return shap_values
            
    except Exception as e:
        logger.error(f"Error in explain_rl_predictions: {str(e)}", exc_info=True)
        return None

def analyze_rl_misclassifications(predictions: np.ndarray, true_labels: np.ndarray, 
                                shap_values: np.ndarray, feature_names: List[str], 
                                save_path: str):
    """
    Analyze misclassified samples for RL model predictions.

    Similar to analyze_binary_misclassifications() but handles multi-class case.
    See binary_xai_utils.py for binary version.

    Args:
        predictions: Model's predicted labels
        true_labels: Ground truth labels  
        shap_values: SHAP values from explain_rl_predictions()
        feature_names: Names of input features
        save_path: Directory to save visualizations
    """
    try:
        # Handle size mismatches
        if len(predictions) > len(shap_values):
            predictions = predictions[:len(shap_values)]
            true_labels = true_labels[:len(shap_values)]
            logger.info(f"Truncated predictions and labels to match SHAP values size: {len(shap_values)}")
        
        misclassified_indices = np.where(predictions != true_labels)[0]
        
        if len(misclassified_indices) == 0:
            logger.info("No misclassifications found in the analyzed subset.")
            return

        # Handle 3D SHAP values for multi-class
        if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # Filter misclassified_indices to only include valid indices
            valid_indices = misclassified_indices[misclassified_indices < len(shap_values)]
            
            if len(valid_indices) == 0:
                logger.info("No valid misclassified samples found in SHAP values range.")
                return
                
            # Average across classes for misclassified samples
            misclassified_shap = np.abs(shap_values[valid_indices]).mean(axis=2)
            
            if len(misclassified_shap) > 0:
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    misclassified_shap,
                    feature_names=feature_names,
                    plot_type="bar",
                    show=False
                )
                plt.title("SHAP Values for RL Misclassified Samples")
                if save_path:
                    misclassified_path = f"{save_path}/rl_misclassified_analysis.png"
                    plt.savefig(misclassified_path, dpi=150, bbox_inches='tight')
                    logger.info(f"RL misclassified analysis plot saved to {misclassified_path}")
                plt.close()
        else:
            logger.warning("Unexpected SHAP values format for misclassification analysis")

    except Exception as e:
        logger.error(f"Error in RL misclassification analysis: {str(e)}", exc_info=True)
        logger.debug(f"SHAP values shape: {shap_values.shape}")
        logger.debug(f"Predictions length: {len(predictions)}")
        logger.debug(f"True labels length: {len(true_labels)}")
        logger.debug(f"Number of misclassified indices: {len(misclassified_indices)}")

def generate_rl_explanation(shap_values: np.ndarray, features: pd.DataFrame, 
                          prediction: int, class_names: List[str], 
                          top_k: int = 5) -> List[Dict]:
    """
    Generate human-readable explanations of RL model decisions.

    Similar to generate_binary_explanation() but handles multi-class predictions.
    See binary_xai_utils.py for binary version.

    Args:
        shap_values: SHAP values from explain_rl_predictions()
        features: Input feature DataFrame
        prediction: Predicted class index
        class_names: List of class names
        top_k: Number of top features to include in explanation

    Returns:
        List[Dict]: Explanations for all classes, sorted by confidence
    """
    try:
        explanations = []
        
        if len(shap_values.shape) == 3:
            sample_idx = 0
            # Calculate total impact across all classes
            total_impact = np.sum([np.sum(np.abs(shap_values[sample_idx, :, i])) 
                                 for i in range(len(class_names))])
            
            # Generate per-class explanations
            for class_idx in range(len(class_names)):
                abs_shap = np.abs(shap_values[sample_idx, :, class_idx])
                feature_importance = pd.DataFrame({
                    'feature': features.columns,
                    'importance': abs_shap.flatten()
                })
                
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                top_features = feature_importance.head(top_k)
                
                class_impact = np.sum(np.abs(shap_values[sample_idx, :, class_idx]))
                confidence = float(class_impact / total_impact if total_impact > 0 else 0)
                
                explanations.append({
                    'class': class_names[class_idx],
                    'is_predicted_class': (class_idx == prediction),
                    'top_features': top_features.to_dict('records'),
                    'confidence': confidence,
                    'raw_impact': float(class_impact)
                })
        
        explanations.sort(key=lambda x: x['confidence'], reverse=True)
        return explanations
        
    except Exception as e:
        logger.error(f"Error in RL explanation generation: {str(e)}")
        return [{
            'class': class_names[prediction],
            'error': str(e),
            'top_features': [],
            'confidence': 0.0
        }]