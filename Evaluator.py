# Evaluator.py

import torch
import numpy as np
import pandas as pd
import os
from agents.base_agent import Agent
from utils.intrusion_detection_env import NetworkClassificationEnv
from utils.q_network import QNetwork  # Ensure this is correctly implemented
from utils.visualizer import plot_confusion_matrix
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import warnings

def evaluate():
    # Suppress FutureWarning for torch.load if necessary
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Load label dictionary
    label_dict_path = "data/mappings/label_dict.json"
    if not os.path.exists(label_dict_path):
        raise FileNotFoundError(f"Label dictionary not found at {label_dict_path}")
        
    with open(label_dict_path, "r") as infile:
        label_dict = json.load(infile)
    label_names = list(label_dict.keys())
    
    # Load test data
    test_data_path = "data/test_df.csv"
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found at {test_data_path}")
    
    test_df = pd.read_csv(test_data_path)
    
    # Initialize environment with a reasonable batch size
    env = NetworkClassificationEnv(test_df, label_dict, batch_size=64)
    
    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(
        state_size=state_size,
        action_size=action_size,
        hidden_layers=[64, 32],
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=0.0,  # No exploration during evaluation
        epsilon_end=0.0,
        epsilon_decay=1.0,
        batch_size=64,
        memory_size=10000,
        device=device
    )
    
    # Load trained model with weights_only=True if supported
    model_path = "models/dqn_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # If weights_only parameter is not supported, proceed without it
        state_dict = torch.load(model_path, map_location=device)
    agent.qnetwork_local.load_state_dict(state_dict)
    agent.qnetwork_local.eval()
    
    # Collect all states and labels
    states, labels = env.reset()
    all_actions = []
    all_labels = []
    done = False
    
    while not done:
        actions = agent.act(states)  # actions should be a list
        # Debugging: Verify the type and content of actions
        print(f"actions: {actions}, type: {type(actions)}, length: {len(actions)}")
        all_actions.extend(actions)
        all_labels.extend(labels)
        next_states, rewards, done, next_labels = env.step(actions)
        states, labels = next_states, next_labels  # Correctly assign next_labels
    
    # Evaluation Metrics
    print("Generating classification report...")
    print(classification_report(all_labels, all_actions, target_names=label_names))
    
    accuracy = accuracy_score(all_labels, all_actions)
    f1 = f1_score(all_labels, all_actions, average='weighted')
    precision = precision_score(all_labels, all_actions, average='weighted')
    recall = recall_score(all_labels, all_actions, average='weighted')
    
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Confusion Matrix
    cm_save_path = "results/confusion_matrix.png"
    plot_confusion_matrix(all_labels, all_actions, classes=label_names, save_path=cm_save_path, normalize=True, title='Normalized Confusion Matrix')
    print(f"Confusion matrix saved to {cm_save_path}")
    
    # Save results to a list of dictionaries
    rows = []
    unique_labels = np.unique(all_labels)
    for label in unique_labels:
        estimated = all_actions.count(label)
        correct = sum(1 for a, l in zip(all_actions, all_labels) if a == l and l == label)
        total = all_labels.count(label)
        acc = (correct / total) * 100 if total > 0 else 0
        rows.append({
            "Estimated": estimated,
            "Correct": correct,
            "Total": total,
            "Accuracy": acc
        })
    
    # Create the DataFrame from the list of dictionaries with label_names as the index
    outputs_df = pd.DataFrame(rows, index=label_names)
    
    # Save to CSV
    outputs_save_path = "results/evaluation_results.csv"
    outputs_df.to_csv(outputs_save_path)
    print(f"Evaluation results saved to {outputs_save_path}")

if __name__ == "__main__":
    evaluate()
