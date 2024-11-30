"""
Trainer module for managing the training and evaluation of reinforcement learning agents.

This module provides functionality to:
1. Train agents over multiple episodes
2. Track training metrics (rewards, losses)
3. Evaluate trained agents
4. Generate performance metrics and probabilities

Classes:
    Trainer: Manages the training and evaluation of RL agents.
"""

# utils/trainer.py

import torch
import numpy as np
import logging
from scipy.special import softmax
from collections import deque
from utils.evaluation import evaluate_rl_agent

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, env, agent):
        """
        Initializes the Trainer with the environment and agent.
        
        Parameters:
        - env (NetworkClassificationEnv): The environment for training.
        - agent (Agent): The RL agent to train.
        """
        self.env = env
        self.agent = agent

    def train(self, num_episodes, print_interval, X_test, y_test):
        """
        Trains the RL agent for a specified number of episodes.

        Parameters:
        - num_episodes (int): Number of training episodes.
        - print_interval (int): Interval at which to print training status.
        - X_test (np.ndarray): Test set features for evaluation.
        - y_test (np.ndarray): Test set labels for evaluation.

        Returns:
        - reward_history (list): History of rewards per episode.
        - loss_history (list): History of losses per training step.
        """
        reward_history = []
        loss_history = []
        recent_rewards = deque(maxlen=10)  # Track last 10 rewards for averaging

        for episode in range(1, num_episodes + 1):
            states, labels = self.env.reset()
            total_reward = 0
            losses = []
            done = False

            while not done:
                actions = self.agent.act(states)
                next_states, rewards, done, next_labels = self.env.step(actions)
                
                # Store experiences and learn
                for state, action, reward, next_state in zip(states, actions, rewards, next_states):
                    loss = self.agent.step(state, action, reward, next_state, done)
                    if loss is not None:
                        losses.append(loss)
                
                total_reward += sum(rewards)
                states = next_states

            # Record metrics
            avg_loss = np.mean(losses) if losses else 0
            reward_history.append(total_reward)
            loss_history.append(avg_loss)
            recent_rewards.append(total_reward)
            avg_reward = np.mean(recent_rewards)

            # Print progress at intervals
            if episode % print_interval == 0:
                y_pred = self.agent.predict(X_test)  # Obtain predictions using the Agent's predict method
                metrics = evaluate_rl_agent(
                    y_true=y_test,                # Ensure y_test is integer-encoded
                    y_pred=y_pred,                # Predicted labels from Agent
                    label_dict_path="processed_data/multi_class_classification/label_dict.json"  # Path to label_dict.json
                )
                logger.info(f"Episode {episode}/{num_episodes} - Metrics: {metrics}")

        return reward_history, loss_history

    def evaluate(self):
        """
        Evaluates the trained agent on the environment.
        
        Returns:
        - y_true (np.ndarray): True labels.
        - y_scores (np.ndarray): Predicted Q-values or probabilities.
        """
        y_true = []
        y_scores = []

        states, labels = self.env.reset()
        # Extract the first (and only) state from the batch
        state = states[0]
        done = False

        while not done:
            # Agent predicts action and Q-values for the single state
            preds, q_values = self.agent.predict_batch([state])  # Pass list of single state

            # Collect Q-values for the current state
            y_scores.append(q_values[0])  # q_values[0] is a 1D array of length num_actions

            # Get the true label from the environment
            true_label = self.env.get_true_label(state)
            y_true.append(true_label)

            # Take action in the environment
            action = preds[0]
            next_states, reward, done, info = self.env.step([action])  # Pass as list containing a single action

            # Extract the next single state from the batch
            state = next_states[0]

        # Convert lists to numpy arrays
        y_true = np.array(y_true)               # Shape: [num_samples]
        y_scores = np.vstack(y_scores)          # Shape: [num_samples, num_actions]

        # Apply softmax to get probabilities
        y_scores = softmax(y_scores, axis=1)    # Shape: [num_samples, num_actions]

        logger.info("Evaluation completed.")
        return y_true, y_scores
