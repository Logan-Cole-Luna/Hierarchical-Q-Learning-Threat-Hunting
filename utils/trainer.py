"""
trainer.py

Defines the Trainer class to manage the training process for high-level and low-level agents 
in an intrusion detection environment. The Trainer class coordinates interactions between 
agents and the environment over multiple episodes.

Classes:
    - Trainer: Orchestrates the training loop, manages agent actions, and updates for each episode.
"""

import numpy as np
import logging
from utils.intrusion_detection_env import IntrusionDetectionEnv
from agents.high_level_agent import HighLevelAgent
from agents.low_level_agent import LowLevelAgent

class Trainer:
    def __init__(self, env, high_agent, low_agent, episodes=1000):
        """
        Initializes the Trainer with environment, agents, and training parameters.

        Parameters:
        -----------
        env : IntrusionDetectionEnv
            The environment where agents interact and receive rewards.
        high_agent : HighLevelAgent
            Agent responsible for high-level actions.
        low_agent : LowLevelAgent
            Agent responsible for low-level actions.
        episodes : int, optional
            Number of training episodes (default is 1000).
        """
        self.env = env
        self.high_agent = high_agent
        self.low_agent = low_agent
        self.episodes = episodes

    def get_confidence(self, q_values, action):
        """
        Calculates the confidence level for a given action based on Q-values.

        Parameters:
        -----------
        q_values : np.ndarray
            Q-values for all actions for the current state.
        action : int
            The action taken.

        Returns:
        --------
        float
            Confidence level between 0 and 1.
        """
        max_q = np.max(q_values)
        sum_q = np.sum(q_values) + 1e-6  # Prevent division by zero
        confidence = max_q / sum_q
        return confidence

    def train(self):
        """
        Executes the training loop, iterating over episodes and updating agents' Q-networks.
        """
        for episode in range(1, self.episodes + 1):
            # Reset environment and obtain initial state
            initial_state = self.env.reset()
            state_high, state_low, high_label, low_label = initial_state

            # Calculate the correct state sizes
            state_size_high = self.env.history_length * self.env.X_high.shape[1] + 1
            state_size_low = self.env.history_length * self.env.X_low.shape[1] + 1

            # Reshape states to match agent's expected input
            state_high = np.reshape(state_high, [1, state_size_high])
            state_low = np.reshape(state_low, [1, state_size_low])

            done = False
            total_reward_high = 0
            total_reward_low = 0

            while not done:
                # High-Level Agent selects an action
                q_values_high = self.high_agent.q_network.predict(state_high)
                action_high = np.argmax(q_values_high[0])
                confidence_high = self.get_confidence(q_values_high[0], action_high)

                # Set goal for Low-Level Agent based on High-Level action
                self.low_agent.set_goal(action_high)

                # Low-Level Agent selects an action
                q_values_low = self.low_agent.q_network.predict(state_low)
                action_low = np.argmax(q_values_low[0])
                confidence_low = self.get_confidence(q_values_low[0], action_low)

                # Execute actions in the environment
                (next_state_high, next_state_low, next_high_label, next_low_label), (reward_high, reward_low), done = self.env.step(
                    action_high, action_low, high_confidence=confidence_high, low_confidence=confidence_low
                )

                # If the episode is not done, reshape the next states
                if next_state_high is not None and next_state_low is not None:
                    next_state_high = np.reshape(next_state_high, [1, state_size_high])
                    next_state_low = np.reshape(next_state_low, [1, state_size_low])
                else:
                    next_state_high, next_state_low = None, None

                # Store experience in agents' replay buffers
                self.high_agent.remember(state_high, action_high, reward_high, next_state_high, done)
                self.low_agent.remember(state_low, action_low, reward_low, next_state_low, done)

                # Update current state for next iteration
                state_high = next_state_high
                state_low = next_state_low

                # Accumulate rewards for this episode
                total_reward_high += reward_high
                total_reward_low += reward_low

            # Train (replay) the agents on sampled experience batches after each episode
            self.high_agent.replay()
            self.low_agent.replay()

            # Display progress at regular intervals
            if episode % 100 == 0 or episode == 1:
                logging.info(f"Episode {episode}/{self.episodes} - High Reward: {total_reward_high}, Low Reward: {total_reward_low}")