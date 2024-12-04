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
from tqdm import tqdm 

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, env, agent, val_env=None):
        """
        Initializes the Trainer with the environment and agent.
        
        Parameters:
        - env (NetworkClassificationEnv): The environment for training.
        - agent (Agent): The RL agent to train.
        - val_env (NetworkClassificationEnv, optional): The environment for validation.
        """
        self.env = env
        self.agent = agent
        self.val_env = val_env  # Add validation environment

    def train(self, num_episodes, print_interval=None, early_stopping_rounds=10):
        """
        Train the agent for the specified number of episodes.
        
        Args:
            num_episodes (int): Number of episodes to train for
            print_interval (int): How often to print progress (every N episodes)
            early_stopping_rounds (int): Number of episodes with no improvement to stop training early
        """
        if print_interval is None:
            print_interval = max(1, num_episodes // 10)

        reward_history = []
        loss_history = []
        recent_rewards = deque(maxlen=10)  # Track last 10 rewards for averaging

        best_loss = float('inf')
        no_improvement = 0
        best_val_loss = float('inf')
        early_stopping_counter = 0

        # Initialize tqdm progress bar
        for episode in tqdm(range(1, num_episodes + 1), desc="Training Episodes"):
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

            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= early_stopping_rounds:
                    logger.info(f"No improvement for {early_stopping_rounds} episodes. Stopping early.")
                    break

            # Validate after each episode if validation environment is provided
            if self.val_env is not None:
                val_loss = self.validate()
                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                if early_stopping_rounds and early_stopping_counter >= early_stopping_rounds:
                    print("Early stopping triggered.")
                    break

            # Print progress at intervals
            if episode % print_interval == 0:
                logger.info(f"Episode {episode}/{num_episodes} - "
                          f"Total Reward: {total_reward:.2f} - "
                          f"Average Reward (last 10): {avg_reward:.2f} - "
                          f"Average Loss: {avg_loss:.4f}")

        return reward_history, loss_history

    def validate(self):
        self.agent.qnetwork_local.eval()
        total_loss = 0
        with torch.no_grad():
            # ...validation loop using self.val_env...
            pass  # Implement validation loss calculation
        self.agent.qnetwork_local.train()
        return total_loss

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
