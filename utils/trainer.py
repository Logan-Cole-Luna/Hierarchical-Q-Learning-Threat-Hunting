# utils/trainer.py

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, env, agent):
        """
        Initializes the Trainer with the environment and agent.
        
        Parameters:
        - env (NetworkClassificationEnv): The environment for training.
        - agent (Agent): The agent to train.
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

        for episode in range(1, num_episodes + 1):
            states, labels = self.env.reset()
            total_reward = 0.0
            losses = []
            done = False

            while not done:
                actions = self.agent.act(states)  # List of actions
                next_states, rewards, done, next_labels = self.env.step(actions)  # Environment step
                
                # Collect experiences
                for state, action, reward, next_state, done_flag in zip(states, actions, rewards, next_states, [done]*len(actions)):
                    self.agent.step(state, action, reward, next_state, done_flag)

                # Update total reward
                total_reward += np.sum(rewards)
                
                # Update states and labels for the next step
                states, labels = next_states, next_labels  # Correctly assign next_labels

            # Log episode results
            reward_history.append(total_reward)
            # For simplicity, collect the latest loss (if available)
            loss = self.agent.learn() if len(self.agent.memory) > self.agent.batch_size else 0.0
            loss_history.append(loss)

            # Logging
            logger.info(f"Episode {episode}/{num_episodes} - Total Reward: {total_reward:.2f} - Average Reward (last 10): {np.mean(reward_history[-10:]):.2f} - Average Loss: {np.mean(loss_history[-10:]):.4f}")

        return reward_history, loss_history
