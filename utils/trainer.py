# utils/trainer.py

import torch
import numpy as np
import logging
from scipy.special import softmax

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

    def train(self, num_episodes):
        """
        Trains the agent over a specified number of episodes.
        
        Parameters:
        - num_episodes (int): Number of training episodes.
        
        Returns:
        - reward_history (list): List of total rewards per episode.
        - loss_history (list): List of loss values per episode.
        """
        reward_history = []
        loss_history = []
        logger.info(f"Starting training for {num_episodes} episodes...")

        for episode in range(1, num_episodes + 1):
            states, labels = self.env.reset()
            total_reward = 0.0
            losses = []
            done = False

            while not done:
                actions = self.agent.act(states)  # List of actions
                next_states, rewards, done, next_labels = self.env.step(actions)  # Env step

                # Collect experiences
                for state, action, reward, next_state, done_flag in zip(states, actions, rewards, next_states, [done]*len(actions)):
                    self.agent.step(state, action, reward, next_state, done_flag)

                # Update total reward
                total_reward += np.sum(rewards)

                # Update states and labels for the next step
                states, labels = next_states, next_labels

            # Log episode results
            reward_history.append(total_reward)
            # For simplicity, collect the latest loss (if available)
            loss = self.agent.learn() if len(self.agent.memory) > self.agent.batch_size else 0.0
            loss_history.append(loss)

            # Logging
            logger.info(f"Episode {episode}/{num_episodes} - Total Reward: {total_reward:.2f} - Average Reward (last 10): {np.mean(reward_history[-10:]):.2f} - Average Loss: {np.mean(loss_history[-10:]):.4f}")

            # Log progress every 10 episodes
            if episode % 10 == 0:
                average_reward = np.mean(reward_history[-10:])
                logger.info(f"Episode {episode}/{num_episodes}, Average Reward: {average_reward:.2f}")

        logger.info("Training completed.")
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
