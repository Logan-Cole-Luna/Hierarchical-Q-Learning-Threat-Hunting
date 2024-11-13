# overfit.py

import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os  # Ensure os is imported
from QNetwork import QNetwork
from ReplayBuffer import ReplayBuffer
from RewardCalculator import RewardCalculator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    
    Parameters:
        seed (int): Seed value.
    """
    import random

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    """
    Main function to execute the training process.
    """
    set_seed(42)  # For reproducibility

    # Load subset data
    subset_dir = './data/subset'
    X_high_subset = np.load(os.path.join(subset_dir, 'X_high_subset.npy'))
    X_low_subset = np.load(os.path.join(subset_dir, 'X_low_subset.npy'))
    y_high_subset = np.load(os.path.join(subset_dir, 'y_high_subset.npy'))  # High-Level labels
    y_low_subset = np.load(os.path.join(subset_dir, 'y_low_subset.npy'))    # Low-Level labels

    # Load mappings
    mappings_dir = './data/mappings'
    with open(os.path.join(mappings_dir, 'category_to_id.json'), 'r') as f:
        category_to_id = json.load(f)
    with open(os.path.join(mappings_dir, 'anomaly_to_id_subset.json'), 'r') as f:
        anomaly_to_id_subset = json.load(f)

    # Create inverse mapping for categories
    category_id_to_label = {v: k for k, v in category_to_id.items()}

    # Verify unique labels
    unique_labels_high = np.unique(y_high_subset)
    unique_labels_low = np.unique(y_low_subset)
    logging.info(f"Unique High-Level labels in subset: {unique_labels_high}")
    logging.info(f"Unique Low-Level labels in subset: {unique_labels_low}")

    # Remap Low-Level labels to a contiguous range [0, N-1]
    unique_low_labels = unique_labels_low.tolist()
    low_label_mapping_subset = {label: idx for idx, label in enumerate(unique_low_labels)}
    y_low_subset_mapped = np.array([low_label_mapping_subset[label] for label in y_low_subset])

    # Define Q-Network parameters
    state_size_high = X_high_subset.shape[1]
    state_size_low = X_low_subset.shape[1]
    action_size_high = len(unique_labels_high)  # Number of high-level classes (including benign)
    action_size_low = len(unique_low_labels)    # Number of low-level classes in subset

    logging.info(f"High-Level Q-Network State Size: {state_size_high}, Action Size: {action_size_high}")
    logging.info(f"Low-Level Q-Network State Size: {state_size_low}, Action Size: {action_size_low}")

    # Initialize Q-Networks
    q_network_high = QNetwork(state_size_high, action_size_high, learning_rate=0.001)
    q_network_low = QNetwork(state_size_low, action_size_low, learning_rate=0.001)

    # Initialize Replay Buffers
    replay_buffer_high = ReplayBuffer(capacity=10000)
    replay_buffer_low = ReplayBuffer(capacity=10000)

    # Define training parameters
    num_epochs = 1000
    batch_size = 64
    gamma = 0.99  # Discount factor
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 500  # Decay rate for epsilon

    epsilon = epsilon_start

    # Initialize RewardCalculator
    benign_id = category_to_id.get('BENIGN', 0)  # Assuming 'BENIGN' represents benign
    reward_calculator = RewardCalculator(category_to_id, anomaly_to_id_subset, benign_id)

    # Training Loop
    logging.info("Starting training...")
    for epoch in range(1, num_epochs + 1):
        for i in range(len(X_high_subset)):
            # Current state
            state_high = X_high_subset[i]  # Shape: (5,)
            state_low = X_low_subset[i]    # Shape: (4,)
            action_true_high = y_high_subset[i]
            action_true_low = y_low_subset_mapped[i]

            # Epsilon-greedy action selection for High-Level
            if np.random.rand() < epsilon:
                action_high = np.random.randint(0, action_size_high)
            else:
                q_values_high, _ = q_network_high.predict(state_high.reshape(1, -1))
                action_high = np.argmax(q_values_high)

            # Predict Low-Level only if High-Level predicts an attack (not benign)
            if action_high != benign_id:
                # Epsilon-greedy action selection for Low-Level
                if np.random.rand() < epsilon:
                    action_low = np.random.randint(0, action_size_low)
                else:
                    q_values_low, _ = q_network_low.predict(state_low.reshape(1, -1))
                    action_low = np.argmax(q_values_low)
            else:
                action_low = None  # No action for Low-Level

            # Determine next state (set to zeros since done=True for all)
            next_state_high = np.zeros(state_high.shape, dtype=float)  # Shape: (5,)
            next_state_low = np.zeros(state_low.shape, dtype=float) if action_low is not None else None
            done = True  # Each data point is an episode

            # Calculate reward
            if action_high != benign_id:
                if action_high == action_true_high:
                    if action_low is not None:
                        if action_low == action_true_low:
                            # Correct High-Level and Low-Level
                            reward_high, reward_low = reward_calculator.calculate_rewards(
                                action_high, action_true_high, 1.0,  # Assuming confidence 1 for correct action
                                action_low, action_true_low, 1.0
                            )
                        else:
                            # Correct High-Level, Incorrect Low-Level
                            reward_high, reward_low = reward_calculator.calculate_rewards(
                                action_high, action_true_high, 1.0,
                                action_low, action_true_low, 0.0
                            )
                    else:
                        reward_high, reward_low = reward_calculator.calculate_rewards(
                            action_high, action_true_high, 1.0,
                            action_low, action_true_low, 0.0
                        )
                else:
                    # High-Level Incorrect, Low-Level not engaged
                    reward_high, reward_low = reward_calculator.calculate_rewards(
                        action_high, action_true_high, 0.0,
                        action_low, action_true_low, 0.0
                    )
            else:
                if action_high == action_true_high:
                    # Correctly predicted benign
                    reward_high, reward_low = reward_calculator.calculate_rewards(
                        action_high, action_true_high, 1.0,
                        action_low, action_true_low, 0.0
                    )
                else:
                    # Incorrectly predicted benign
                    reward_high, reward_low = reward_calculator.calculate_rewards(
                        action_high, action_true_high, 0.0,
                        action_low, action_true_low, 0.0
                    )
            
            # Add to replay buffers
            replay_buffer_high.add(state_high, action_high, reward_high, next_state_high, done)
            if action_low is not None:
                replay_buffer_low.add(state_low, action_low, reward_low, next_state_low, done)

        # Decay epsilon
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * epoch / epsilon_decay)

        # Sample and train High-Level Q-Network
        if len(replay_buffer_high) > batch_size:
            states_high, actions_high, rewards_high, next_states_high, dones_high = replay_buffer_high.sample(batch_size)
            
            # Predict Q-values for next states
            q_values_next_high, _ = q_network_high.predict(next_states_high)
            
            # Convert dones_high from boolean to float (1.0 if done, 0.0 otherwise)
            dones_high_float = dones_high.astype(float)

            # Compute target Q-values
            targets_high = rewards_high + gamma * np.max(q_values_next_high, axis=1) * (1.0 - dones_high_float)

            # Current Q-values
            q_values_current_high, _ = q_network_high.predict(states_high)

            # Update only the actions taken
            q_values_current_high[np.arange(batch_size), actions_high] = targets_high

            # Train on batch
            loss_high = q_network_high.train_on_batch(states_high, q_values_current_high)
        else:
            loss_high = None

        # Sample and train Low-Level Q-Network
        if len(replay_buffer_low) > batch_size:
            states_low, actions_low, rewards_low, next_states_low, dones_low = replay_buffer_low.sample(batch_size)
            # Predict Q-values for next states
            q_values_next_low, _ = q_network_low.predict(next_states_low)
            # Convert dones_low from boolean to float
            dones_low_float = dones_low.astype(float)
            # Compute target Q-values
            targets_low = rewards_low + gamma * np.max(q_values_next_low, axis=1) * (1.0 - dones_low_float)
            # Current Q-values
            q_values_current_low, _ = q_network_low.predict(states_low)
            # Update only the actions taken
            q_values_current_low[np.arange(batch_size), actions_low] = targets_low
            # Train on batch
            loss_low = q_network_low.train_on_batch(states_low, q_values_current_low)
        else:
            loss_low = None

        # Logging
        if epoch % 100 == 0 or epoch == 1:
            # Define formatted loss strings
            high_loss_str = f"{loss_high:.4f}" if loss_high is not None else "N/A"
            low_loss_str = f"{loss_low:.4f}" if loss_low is not None else "N/A"

            logging.info(f"Epoch {epoch}/{num_epochs} - Epsilon: {epsilon:.4f} - High-Level Loss: {high_loss_str}, Low-Level Loss: {low_loss_str}")

    logging.info("Training completed.")

    # Evaluation after training
    logging.info("Evaluating models...")

    # Predict High-Level
    q_values_high, probs_high = q_network_high.predict(X_high_subset)
    predictions_high = np.argmax(q_values_high, axis=1)

    # Predict Low-Level
    q_values_low, probs_low = q_network_low.predict(X_low_subset)
    predictions_low = np.argmax(q_values_low, axis=1)

    # Calculate accuracy
    accuracy_high = np.mean(predictions_high == y_high_subset)
    accuracy_low = np.mean(predictions_low == y_low_subset_mapped)

    logging.info(f"High-Level Q-Network Accuracy on Subset: {accuracy_high * 100:.2f}%")
    logging.info(f"Low-Level Q-Network Accuracy on Subset: {accuracy_low * 100:.2f}%")

    # Detailed classification reports
    # Inverse mapping for readable labels
    category_id_to_label = {v: k for k, v in category_to_id.items()}
    low_label_id_to_label = {v: category_id_to_label.get(k, f"Unknown_{k}") for k, v in low_label_mapping_subset.items()}

    # Define target names for classification report
    target_names_high = [category_id_to_label.get(i, f"Unknown_{i}") for i in unique_labels_high]
    target_names_low = [low_label_id_to_label.get(i, f"Unknown_{i}") for i in range(action_size_low)]

    print("\nClassification Report for High-Level Q-Network:")
    print(classification_report(y_high_subset, predictions_high, target_names=target_names_high, zero_division=0))

    print("Classification Report for Low-Level Q-Network:")
    print(classification_report(y_low_subset_mapped, predictions_low, target_names=target_names_low, zero_division=0))

    # Plot Confusion Matrix for High-Level Q-Network
    cm_high = confusion_matrix(y_high_subset, predictions_high)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_high, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names_high, yticklabels=target_names_high)
    plt.title('Confusion Matrix - High-Level Q-Network')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Plot Confusion Matrix for Low-Level Q-Network
    cm_low = confusion_matrix(y_low_subset_mapped, predictions_low)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_low, annot=True, fmt='d', cmap='Greens',
                xticklabels=target_names_low, yticklabels=target_names_low)
    plt.title('Confusion Matrix - Low-Level Q-Network')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Optional: ROC Curves for High-Level Q-Network
    y_high_binarized = label_binarize(y_high_subset, classes=unique_labels_high)
    n_classes_high = y_high_binarized.shape[1]

    from sklearn.metrics import roc_curve, auc

    fpr_high = dict()
    tpr_high = dict()
    roc_auc_high = dict()
    for i in range(n_classes_high):
        fpr_high[i], tpr_high[i], _ = roc_curve(y_high_binarized[:, i], q_values_high[:, i])
        roc_auc_high[i] = auc(fpr_high[i], tpr_high[i])

    plt.figure(figsize=(10, 8))
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'red']
    for i, color in zip(range(n_classes_high), colors):
        plt.plot(fpr_high[i], tpr_high[i], color=color, lw=2,
                 label=f'ROC curve of {target_names_high[i]} (area = {roc_auc_high[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - High-Level Q-Network')
    plt.legend(loc="lower right")
    plt.show()

    # Optional: ROC Curves for Low-Level Q-Network
    y_low_binarized = label_binarize(y_low_subset_mapped, classes=np.arange(action_size_low))
    n_classes_low = y_low_binarized.shape[1]

    fpr_low = dict()
    tpr_low = dict()
    roc_auc_low = dict()
    for i in range(n_classes_low):
        fpr_low[i], tpr_low[i], _ = roc_curve(y_low_binarized[:, i], q_values_low[:, i])
        roc_auc_low[i] = auc(fpr_low[i], tpr_low[i])

    plt.figure(figsize=(10, 8))
    for i, color in zip(range(n_classes_low), colors):
        plt.plot(fpr_low[i], tpr_low[i], color=color, lw=2,
                 label=f'ROC curve of {target_names_low[i]} (area = {roc_auc_low[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Low-Level Q-Network')
    plt.legend(loc="lower right")
    plt.show()

    # Optional: Gradient Flow Monitoring
    # Uncomment the following lines if you wish to inspect gradients

    '''
    for name, param in q_network_high.named_parameters():
        if param.grad is not None:
            logging.debug(f"High-Level {name} gradient mean: {param.grad.mean():.4f}, std: {param.grad.std():.4f}")
        else:
            logging.debug(f"High-Level {name} has no gradient.")

    for name, param in q_network_low.named_parameters():
        if param.grad is not None:
            logging.debug(f"Low-Level {name} gradient mean: {param.grad.mean():.4f}, std: {param.grad.std():.4f}")
        else:
            logging.debug(f"Low-Level {name} has no gradient.")
    '''


if __name__ == "__main__":
    main()
