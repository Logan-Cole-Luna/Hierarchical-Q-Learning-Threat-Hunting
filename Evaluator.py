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
    with open("data/mappings/label_dict.json", "r") as infile:
        label_dict = json.load(infile)
    label_names = list(label_dict.keys())
    
    # Load test data
    test_df = pd.read_csv("data/test_df.csv")
    
    # Initialize environment with batch_size=1 for simplicity
    env = NetworkClassificationEnv(test_df, label_dict, batch_size=1)
    
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
    try:
        state_dict = torch.load("models/dqn_model.pth", map_location=device, weights_only=True)
    except TypeError:
        # If weights_only parameter is not supported, proceed without it
        state_dict = torch.load("models/dqn_model.pth", map_location=device)
    agent.qnetwork_local.load_state_dict(state_dict)
    agent.qnetwork_local.eval()
    
    # Collect all states and labels
    states, labels = env.reset()
    all_actions = []
    all_labels = []
    done = False
    
    while not done:
        # Since batch_size=1, states and labels are single samples
        actions = agent.act(states)  # actions is a list with one element
        all_actions.extend(actions)  # Extend the list with the action(s)
        all_labels.extend(labels)    # Extend the list with the label(s)
        next_states, rewards, done, _ = env.step(actions)
        states = next_states
    
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
    
    # Save results to a DataFrame
    outputs_df = pd.DataFrame(columns=["Estimated", "Correct", "Total", "Accuracy"])
    unique_labels = np.unique(all_labels)
    for label in unique_labels:
        estimated = all_actions.count(label)
        correct = sum(1 for a, l in zip(all_actions, all_labels) if a == l and l == label)
        total = all_labels.count(label)
        acc = (correct / total) * 100 if total > 0 else 0
        outputs_df = outputs_df.append({
            "Estimated": estimated,
            "Correct": correct,
            "Total": total,
            "Accuracy": acc
        }, ignore_index=True)
    
    outputs_df.index = label_names
    outputs_save_path = "results/evaluation_results.csv"
    outputs_df.to_csv(outputs_save_path)
    print(f"Evaluation results saved to {outputs_save_path}")

if __name__ == "__main__":
    evaluate()
