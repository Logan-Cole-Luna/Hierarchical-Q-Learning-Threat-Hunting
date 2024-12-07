import shap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
import pandas as pd
import xgboost as xgb
from typing import List, Dict

logger = logging.getLogger(__name__)

# Ensure matplotlib is using a suitable backend
matplotlib.use('TkAgg')  # Or 'Agg' if running without a display server

def explain_binary_predictions(model, data: pd.DataFrame, feature_names: List[str], 
                            save_path: str = None, max_samples: int = 1000):
    """Generate SHAP explanations for binary classifier predictions."""
    try:
        if len(data) > max_samples:
            data = data.sample(n=max_samples, random_state=42)
            logger.info(f"Subsampled data to {max_samples} samples for SHAP analysis")

        # For XGBoost binary classifier
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data.values)
        
        # Generate visualizations
        if save_path:
            logger.debug("Generating SHAP summary plot for binary classification.")
            plt.figure(figsize=(12, 8))
            if isinstance(shap_values, list) and len(shap_values) > 1:
                logger.debug(f"shap_values is a list with {len(shap_values)} elements.")
                # Show SHAP values for positive class (Malicious)
                shap.summary_plot(
                    shap_values[1],
                    data.values,
                    feature_names=feature_names,
                    plot_type="bar"  # Removed show=False
                )
                plt.title("Binary Classification SHAP Summary")
                plt.tight_layout()
                summary_path = f"{save_path}/binary_shap_summary.png"
                plt.savefig(summary_path, dpi=150, bbox_inches='tight')
                logger.info(f"Binary SHAP summary plot saved to {summary_path}")
                plt.show()  # Add show call
                plt.close()
            else:
                logger.warning("shap_values is not a list or does not have expected elements.")
        else:
            logger.warning("Save path not provided. Skipping saving SHAP summary plot.")
        
        return shap_values
        
    except Exception as e:
        logger.error(f"Error generating binary SHAP explanations: {str(e)}", exc_info=True)
        return None

def analyze_binary_misclassifications(predictions: np.ndarray, true_labels: np.ndarray, 
                                    shap_values: np.ndarray, feature_names: List[str], 
                                    save_path: str):
    """Analyze misclassified samples for binary classification."""
    try:
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        misclassified_indices = np.where(predictions != true_labels)[0]
        
        if len(misclassified_indices) == 0:
            logger.info("No misclassifications found.")
            return

        # For binary classification, use positive class SHAP values
        if isinstance(shap_values, list):
            misclassified_shap = shap_values[1][misclassified_indices]
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
            plt.title("SHAP Values for Binary Misclassified Samples")
            if save_path:
                misclassified_path = f"{save_path}/binary_misclassified_analysis.png"
                plt.savefig(misclassified_path, dpi=150, bbox_inches='tight')
                logger.info(f"Binary misclassified analysis plot saved to {misclassified_path}")
            else:
                logger.warning("Save path not provided. Skipping saving misclassified plot.")
            plt.show()  # Add show call
            plt.close()
        else:
            logger.info("No misclassified SHAP values to plot.")
    except Exception as e:
        logger.error(f"Error in binary misclassification analysis: {str(e)}", exc_info=True)

def generate_binary_explanation(shap_values: np.ndarray, features: pd.DataFrame, 
                              prediction: int, class_names: List[str], 
                              top_k: int = 5) -> List[Dict]:
    """Generate explanations for binary classification."""
    try:
        explanations = []
        sample_idx = 0
        
        for class_idx, class_name in enumerate(class_names):
            if isinstance(shap_values, list):
                # Use appropriate class SHAP values
                abs_shap = np.abs(shap_values[class_idx][sample_idx])
            else:
                abs_shap = np.abs(shap_values[sample_idx])
                if class_idx == 1:  # For malicious class
                    abs_shap = -abs_shap  # Negate values for opposite class
            
            feature_importance = pd.DataFrame({
                'feature': features.columns,
                'importance': abs_shap
            })
            
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            top_features = feature_importance.head(top_k)
            confidence = float(np.sum(np.abs(abs_shap)))
            
            explanations.append({
                'class': class_name,
                'is_predicted_class': (class_idx == prediction),
                'top_features': top_features.to_dict('records'),
                'confidence': confidence
            })
        
        explanations.sort(key=lambda x: x['confidence'], reverse=True)
        return explanations
        
    except Exception as e:
        logger.error(f"Error in binary explanation generation: {str(e)}")
        return [{
            'class': class_names[prediction],
            'is_predicted_class': True,
            'error': str(e),
            'top_features': [],
            'confidence': 0.0
        }]