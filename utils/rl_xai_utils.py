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
            background = shap.kmeans(data_values[:min(30, len(data))], k=3)
            
            def model_predict(x):
                outputs = model.run(None, {model.get_inputs()[0].name: x})[0]
                exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
                return exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
            
            explainer = shap.KernelExplainer(
                model_predict,
                background,
                link="identity",
                feature_perturbation="interventional"
            )
            
            shap_values = explainer.shap_values(
                data_values[:min(50, len(data))],
                nsamples=50
            )
            
            if save_path:
                logger.debug("Generating SHAP summary plot for RL model.")
                plt.figure(figsize=(12, 8))
                if isinstance(shap_values, list) and len(shap_values) > 0:
                    logger.debug(f"shap_values is a list with {len(shap_values)} elements.")
                    shap.summary_plot(
                        shap_values[0],  # Use first class's SHAP values
                        data_values[:min(500, len(data))],
                        feature_names=feature_names,
                        plot_type="bar"  # Add plot_type
                    )
                    plt.title("RL Model SHAP Summary")  # Add title
                    plt.tight_layout()
                    summary_path = f"{save_path}/rl_shap_summary.png"
                    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
                    plt.show()  # Add show call
                    plt.close()
                    logger.info(f"RL SHAP summary plot saved to {summary_path}")
                else:
                    logger.warning("shap_values is not a non-empty list. Cannot generate SHAP summary plot.")
            else:
                logger.warning("Save path not provided. Skipping saving SHAP summary plot.")
            
            return shap_values
            
    except Exception as e:
        logger.error(f"Error generating RL SHAP explanations: {str(e)}", exc_info=True)
        return None

def analyze_rl_misclassifications(predictions: np.ndarray, true_labels: np.ndarray, 
                                shap_values: np.ndarray, feature_names: List[str], 
                                save_path: str):
    """Analyze misclassified samples for RL model."""
    try:
        misclassified_indices = np.where(predictions != true_labels)[0]
        
        if len(misclassified_indices) == 0:
            logger.info("No misclassifications found.")
            return

        if isinstance(shap_values, list):
            misclassified_shap = np.mean([np.abs(sv[misclassified_indices]) for sv in shap_values], axis=0)
        else:
            misclassified_shap = shap_values[misclassified_indices]
        
        if len(misclassified_shap) > 0 and misclassified_shap is not None:
            logger.debug(f"Number of misclassified samples: {len(misclassified_indices)}")
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                misclassified_shap,
                feature_names=feature_names,
                plot_size=(12, 8)  # Removed show=False
            )
            plt.title("SHAP Values for RL Misclassified Samples")
            if save_path:
                misclassified_path = f"{save_path}/rl_misclassified_analysis.png"
                plt.savefig(misclassified_path, dpi=150, bbox_inches='tight')
                logger.info(f"RL misclassified analysis plot saved to {misclassified_path}")
            else:
                logger.warning("Save path not provided. Skipping saving misclassified plot.")
            plt.show()  # Add show call
            plt.close()
        else:
            logger.info("No misclassified SHAP values to plot.")
            
    except Exception as e:
        logger.error(f"Error in RL misclassification analysis: {str(e)}", exc_info=True)

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