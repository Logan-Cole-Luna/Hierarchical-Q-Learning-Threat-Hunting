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
    """Generate SHAP explanations for RL model predictions."""
    try:
        if len(data) > max_samples:
            data = data.sample(n=max_samples, random_state=42)
            logger.info(f"Subsampled data to {max_samples} samples for SHAP analysis")
        
        if isinstance(model, ort.InferenceSession):
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
            
            # Test model prediction
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
            
            # Generate SHAP values with smaller nsamples for testing
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
            
            if save_path and shap_values is not None:
                try:
                    plt.figure(figsize=(12, 8))
                    
                    # Handle 3D SHAP values for multi-class
                    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                        # Average absolute SHAP values across classes
                        avg_shap = np.abs(shap_values).mean(axis=2)
                        
                        shap.summary_plot(
                            avg_shap,
                            data_values[:min(10, len(data))],
                            feature_names=feature_names,
                            plot_type="bar",
                            show=False
                        )
                        plt.title("RL Model SHAP Summary (Averaged Across Classes)")
                        plt.tight_layout()
                        summary_path = f"{save_path}/rl_shap_summary.png"
                        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        logger.info(f"RL SHAP summary plot saved to {summary_path}")
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
    """Analyze misclassified samples for RL model."""
    try:
        # Ensure we only look at indices that exist in our shap_values
        if len(predictions) > len(shap_values):
            predictions = predictions[:len(shap_values)]
            true_labels = true_labels[:len(shap_values)]
            logger.info(f"Truncated predictions and labels to match SHAP values size: {len(shap_values)}")
        
        misclassified_indices = np.where(predictions != true_labels)[0]
        
        if len(misclassified_indices) == 0:
            logger.info("No misclassifications found in the analyzed subset.")
            return

        # Handle 3D SHAP values
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
            logger.warning(f"Unexpected SHAP values format for misclassification analysis")

    except Exception as e:
        logger.error(f"Error in RL misclassification analysis: {str(e)}", exc_info=True)
        logger.debug(f"SHAP values shape: {shap_values.shape}")
        logger.debug(f"Predictions length: {len(predictions)}")
        logger.debug(f"True labels length: {len(true_labels)}")
        logger.debug(f"Number of misclassified indices: {len(misclassified_indices)}")

def generate_rl_explanation(shap_values: np.ndarray, features: pd.DataFrame, 
                          prediction: int, class_names: List[str], 
                          top_k: int = 5) -> List[Dict]:
    """Generate explanations for RL model predictions."""
    try:
        explanations = []
        
        if len(shap_values.shape) == 3:
            sample_idx = 0
            total_impact = np.sum([np.sum(np.abs(shap_values[sample_idx, :, i])) 
                                 for i in range(len(class_names))])
            
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