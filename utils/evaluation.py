# utils/evaluation.py

import torch
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from utils.visualizer import plot_confusion_matrix, plot_roc_curves
import logging
import time
from scipy.stats import entropy
import shap

logger = logging.getLogger(__name__)

def evaluate_agent(agent, env, label_dict, test_df=None, save_confusion_matrix=True, save_roc_curves=True, save_path='results'):
    """
    Evaluates the trained agent on the test dataset and generates evaluation metrics and visualizations.

    Parameters:
    - agent (Agent): The trained agent.
    - env (NetworkClassificationEnv): The environment to interact with.
    - label_dict (dict): Dictionary mapping label names to indices.
    - test_df (pd.DataFrame, optional): The test dataframe. If None, it will be loaded from env.
    - save_confusion_matrix (bool): Whether to save the confusion matrix plot.
    - save_roc_curves (bool): Whether to save the ROC curves plot.
    - save_path (str): Directory path to save the results.

    Returns:
    - metrics (dict): Dictionary containing evaluation metrics.
    """
    start_time = time.time()
    # Ensure save_path exists
    os.makedirs(save_path, exist_ok=True)

    # If test_df is provided, use it; otherwise, assume env has the test data
    if test_df is None:
        # You might need to adjust this based on how your environment is set up
        raise NotImplementedError("Please provide the test_df parameter.")

    # Initialize lists to collect all actions and true labels
    all_actions = []
    all_labels = []

    # Set agent to evaluation mode
    device = agent.device
    agent.qnetwork_local.eval()

    logger.info("Starting agent evaluation...")

    try:
        episode_metrics = []
        prediction_uncertainties = []
        eval_losses = []
        
        states, labels = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Get actions and Q-values from the agent
            actions, q_values = agent.act(states, return_q_values=True)
            
            # Calculate prediction uncertainty using entropy
            uncertainties = np.array([entropy(q_dist) for q_dist in q_values])
            prediction_uncertainties.extend(uncertainties)
            
            # Calculate loss for monitoring
            loss = agent.compute_loss(states, actions, labels)
            eval_losses.append(loss.item())
            
            # Store episode-level metrics
            episode_metrics.append({
                'actions': actions,
                'rewards': rewards,
                'q_values': q_values,
                'uncertainty': uncertainties
            })
            
            # Take a step in the environment
            next_states, rewards, done, next_labels = env.step(actions)
            states, labels = next_states, next_labels

    except Exception as e:
        logger.exception("An error occurred during evaluation.")
        raise e

    # Convert lists to numpy arrays for metric computations
    all_actions = np.array(all_actions)
    all_labels = np.array(all_labels)

    logger.info("Generating evaluation metrics...")

    # Classification Report
    class_report = classification_report(all_labels, all_actions, target_names=label_dict.keys(), output_dict=True)
    logger.info("Classification Report:")
    print(classification_report(all_labels, all_actions, target_names=label_dict.keys()))

    # Overall Metrics
    accuracy = accuracy_score(all_labels, all_actions)
    f1 = f1_score(all_labels, all_actions, average='weighted')
    precision = precision_score(all_labels, all_actions, average='weighted')
    recall = recall_score(all_labels, all_actions, average='weighted')

    logger.info(f"Accuracy: {accuracy*100:.2f}%")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_actions)
    if save_confusion_matrix:
        cm_save_path = os.path.join(save_path, "confusion_matrix.png")
        plot_confusion_matrix(all_labels, all_actions, classes=list(label_dict.keys()), save_path=cm_save_path, normalize=True, title='Normalized Confusion Matrix')
        logger.info(f"Confusion matrix saved to {cm_save_path}")

    # ROC Curves (For Multiclass)
    if save_roc_curves:
        # Binarize the labels for ROC computation
        unique_labels = list(label_dict.values())
        label_names = list(label_dict.keys())
        all_labels_binarized = label_binarize(all_labels, classes=unique_labels)
        all_actions_binarized = label_binarize(all_actions, classes=unique_labels)
        n_classes = all_labels_binarized.shape[1]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(all_labels_binarized[:, i], all_actions_binarized[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curves
        roc_save_path = os.path.join(save_path, "roc_curves.png")
        plot_roc_curves(fpr, tpr, roc_auc, classes=label_names, save_path=roc_save_path)
        logger.info(f"ROC curves saved to {roc_save_path}")

    # Save Metrics to CSV
    metrics_df = pd.DataFrame(class_report).transpose()
    metrics_save_path = os.path.join(save_path, "evaluation_metrics.csv")
    metrics_df.to_csv(metrics_save_path)
    logger.info(f"Evaluation metrics saved to {metrics_save_path}")

    # Add SHAP values for interpretability
    background = states[:100]  # Sample background data
    explainer = shap.DeepExplainer(agent.qnetwork_local, background)
    shap_values = explainer.shap_values(states)
    
    # Reset agent to training mode if needed
    agent.qnetwork_local.train()

    # Return metrics dictionary
    metrics = {
        "classification_report": class_report,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "eval_time": time.time() - start_time,
        "mean_uncertainty": np.mean(prediction_uncertainties),
        "mean_eval_loss": np.mean(eval_losses),
        "episode_metrics": episode_metrics,
        "shap_values": shap_values
    }

    return metrics
