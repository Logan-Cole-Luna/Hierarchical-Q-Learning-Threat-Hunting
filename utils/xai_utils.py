import shap
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
import onnxruntime as ort
from typing import List, Dict
import pandas as pd

logger = logging.getLogger(__name__)

def explain_predictions(model, data: pd.DataFrame, feature_names: List[str], 
                       num_samples: int = 100, save_path: str = None,
                       max_samples: int = 1000):
    """
    Generate SHAP explanations for model predictions with optimizations.
    
    Args:
        model: Trained model (XGBoost or ONNX session)
        data: Input data to explain
        feature_names: List of feature names
        num_samples: Number of background samples for SHAP
        save_path: Path to save visualization
        max_samples: Maximum number of samples to use for SHAP analysis
    """
    try:
        # Subsample data if it's too large
        if len(data) > max_samples:
            data = data.sample(n=max_samples, random_state=42)
            logger.info(f"Subsampled data to {max_samples} samples for SHAP analysis")
        
        if isinstance(model, ort.InferenceSession):
            # Ensure minimum sample size
            min_samples = max(50, len(feature_names) + 10)  # Ensure more samples than features
            data_subset_size = min(len(data), max(min_samples, max_samples))
            data = data.sample(n=data_subset_size, random_state=42)
            data_values = data.values.astype(np.float32)
            
            # Increase background samples
            background = shap.kmeans(data_values[:min(30, len(data))], k=3)
            
            def model_predict(x):
                outputs = model.run(None, {model.get_inputs()[0].name: x})[0]
                # Current normalization might be squashing values too much
                exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
                return exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
            
            # Configure KernelExplainer with more stable parameters
            explainer = shap.KernelExplainer(
                model_predict,
                background,
                link="identity",
                feature_perturbation="interventional",
                l1_reg="auto",
                eps=1e-2,
                max_iter=10,
                batch_size=10,
                silent=True  # Add silent parameter to reduce logging
            )
            
            # Suppress detailed calculation logs
            import logging
            logging.getLogger('shap').setLevel(logging.WARNING)
            
            # Calculate SHAP values with optimized parameters
            shap_values = explainer.shap_values(
                data_values[:min(50, len(data))],
                nsamples=50,
                l1_reg="auto",
                eps=1e-2,
                silent=True  # Add silent parameter here too
            )
            
            # Generate visualizations
            if save_path:
                plt.figure(figsize=(12, 8))
                if isinstance(shap_values, list):
                    # For multi-class, use first class's SHAP values
                    shap_data = shap_values[0] if isinstance(shap_values, list) else shap_values
                    shap.summary_plot(
                        shap_data,
                        data_values[:min(500, len(data))],
                        feature_names=feature_names,
                        show=False
                    )
                plt.tight_layout()
                plt.savefig(f"{save_path}/shap_summary.png", dpi=150)
                plt.close()
            
            return shap_values
            
        else:
            # For tree models, use the more efficient TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data.values)
        
        # Generate visualizations
        if save_path:
            plt.figure(figsize=(12, 8))
            if isinstance(shap_values, list):
                # For multi-class output
                shap.summary_plot(shap_values[0], data.values, feature_names=feature_names, 
                                show=False, plot_size=(12, 8))
            else:
                # For single-class output
                shap.summary_plot(shap_values, data.values, feature_names=feature_names, 
                                show=False, plot_size=(12, 8))
            plt.tight_layout()
            plt.savefig(f"{save_path}/shap_summary.png", dpi=150)
            plt.close()
            
        return shap_values
        
    except Exception as e:
        logger.error(f"Error generating SHAP explanations: {str(e)}")
        return None

def analyze_misclassifications(predictions: np.ndarray, true_labels: np.ndarray, 
                             shap_values: np.ndarray, feature_names: List[str], 
                             save_path: str):
    """Analyze misclassified samples using SHAP values."""
    try:
        # Ensure all arrays are the same length by truncating to shortest
        min_length = min(len(predictions), len(true_labels))
        if isinstance(shap_values, list):
            min_length = min(min_length, len(shap_values[0]))
        else:
            min_length = min(min_length, len(shap_values))
            
        predictions = np.array(predictions[:min_length])
        true_labels = np.array(true_labels[:min_length])
        
        misclassified_indices = np.where(predictions != true_labels)[0]
        
        if len(misclassified_indices) == 0:
            logger.info("No misclassifications found.")
            return

        # Suppress numpy warnings about array computation
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Handle SHAP values based on type
            if isinstance(shap_values, list):
                shap_values = [sv[:min_length] for sv in shap_values]
            else:
                if len(shap_values.shape) == 3:
                    shap_values = shap_values[:min_length]
                else:
                    shap_values = shap_values[:min_length]

            # Convert shap_values to array if it's a list
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)

            # Handle different SHAP value shapes
            if isinstance(shap_values, np.ndarray):
                if len(shap_values.shape) == 3:
                    # Multi-class case: (samples, features, classes)
                    misclassified_shap = shap_values[misclassified_indices]
                    # Average across classes
                    misclassified_shap = np.mean(np.abs(misclassified_shap), axis=2)
                elif len(shap_values.shape) == 2:
                    # Binary case: (samples, features)
                    misclassified_shap = shap_values[misclassified_indices]
                else:
                    raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")
            else:
                raise ValueError(f"Unexpected SHAP values type: {type(shap_values)}")
        
        if len(misclassified_shap) > 0:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                misclassified_shap,
                feature_names=feature_names,
                show=False,
                plot_size=(12, 8)
            )
            plt.title("SHAP Values for Misclassified Samples")
            if save_path:
                plt.savefig(f"{save_path}/misclassified_analysis.png", dpi=150, bbox_inches='tight')
            plt.close()
            
    except Exception as e:
        logger.error(f"Error in analyze_misclassifications: {str(e)}")
        logger.debug(f"Predictions shape: {predictions.shape}")
        logger.debug(f"True labels shape: {true_labels.shape}")
        logger.debug(f"SHAP values type: {type(shap_values)}")
        if isinstance(shap_values, (list, np.ndarray)):
            logger.debug(f"SHAP values shape: {np.array(shap_values).shape}")

def generate_decision_explanation(shap_values: np.ndarray, features: pd.DataFrame, 
                                prediction: int, class_names: List[str], 
                                top_k: int = 5) -> List[Dict]:
    """Generate human-readable explanations for all classes."""
    try:
        explanations = []
        
        # Handle 3D SHAP values (samples, features, classes)
        if len(shap_values.shape) == 3:
            sample_idx = 0  # Use first sample
            
            # Calculate total absolute SHAP values across all classes for normalization
            total_impact = np.sum([np.sum(np.abs(shap_values[sample_idx, :, i])) 
                                 for i in range(len(class_names))])
            
            # Generate explanation for each class
            for class_idx in range(len(class_names)):
                abs_shap = np.abs(shap_values[sample_idx, :, class_idx])
                
                # Create feature importance DataFrame for this class
                feature_importance = pd.DataFrame({
                    'feature': features.columns,
                    'importance': abs_shap.flatten()
                })
                
                # Sort and get top features
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                top_features = feature_importance.head(top_k)
                
                # Calculate normalized confidence score for this class
                class_impact = np.sum(np.abs(shap_values[sample_idx, :, class_idx]))
                confidence = float(class_impact / total_impact if total_impact > 0 else 0)
                
                explanations.append({
                    'class': class_names[class_idx],
                    'is_predicted_class': (class_idx == prediction),
                    'top_features': top_features.to_dict('records'),
                    'confidence': confidence,
                    'raw_impact': float(class_impact)
                })
        else:
            # For 2D SHAP values, handle each class separately
            for class_idx in range(len(class_names)):
                abs_shap = np.abs(shap_values[:, class_idx])
                
                feature_importance = pd.DataFrame({
                    'feature': features.columns,
                    'importance': abs_shap
                })
                
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                top_features = feature_importance.head(top_k)
                
                confidence = float(np.mean(np.abs(shap_values[:, class_idx])))
                
                explanations.append({
                    'class': class_names[class_idx],
                    'is_predicted_class': (class_idx == prediction),
                    'top_features': top_features.to_dict('records'),
                    'confidence': confidence
                })
        
        # Sort explanations by confidence
        explanations.sort(key=lambda x: x['confidence'], reverse=True)
        return explanations
        
    except Exception as e:
        logger.error(f"Error in generate_decision_explanation: {str(e)}")
        return [{
            'class': class_names[prediction],
            'error': str(e),
            'top_features': [],
            'confidence': 0.0
        }]