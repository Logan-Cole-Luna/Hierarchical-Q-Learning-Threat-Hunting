"""
Trainer.py

Defines the `Trainer` class to manage the training process for high-level and low-level agents 
in an intrusion detection environment. The `Trainer` class coordinates interactions between 
agents and the environment over multiple episodes.

Classes:
    - Trainer: Orchestrates the training loop, manages agent actions, and updates for each episode.
"""

import numpy as np
import logging
from utils.intrusion_detection_env import IntrusionDetectionEnv
from agents.high_level_agent import HighLevelAgent
from agents.low_level_agent import LowLevelAgent
from utils.reward_calculator import RewardCalculator

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

    def train(self):
        """
        Executes the training loop, iterating over episodes and updating agents' Q-networks.
        """
        for episode in range(1, self.episodes + 1):
            state_high, state_low, high_label, low_label = self.env.reset()
            state_high = np.reshape(state_high, [1, self.high_agent.state_size + 1])
            state_low = np.reshape(state_low, [1, self.low_agent.state_size + 1])
            done = False
            total_reward_high = 0
            total_reward_low = 0

            while not done:
                # Agents select actions based on current state
                action_high = self.high_agent.act(state_high)
                self.low_agent.set_goal(action_high)
                action_low = self.low_agent.act(state_low)
                
                # Environment step: execute actions and observe next state, rewards, and done flag
                (next_state_high, next_state_low, next_high_label, next_low_label), (reward_high, reward_low), done = self.env.step(action_high, action_low)
                
                # Reshape next states for consistency
                if not done:
                    next_state_high = np.reshape(next_state_high, [1, self.high_agent.state_size + 1])
                    next_state_low = np.reshape(next_state_low, [1, self.low_agent.state_size + 1])

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
