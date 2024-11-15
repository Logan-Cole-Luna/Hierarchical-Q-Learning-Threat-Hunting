# utils/trainer.py

import torch
import logging
from utils.visualizer import Visuals

class Trainer:
    def __init__(self, env, high_agent, low_agent, episodes=1000, target_update_freq=10):
        self.env = env
        self.high_agent = high_agent
        self.low_agent = low_agent
        self.episodes = episodes
        self.target_update_freq = target_update_freq
        self.loss_high_history = []
        self.loss_low_history = []

    def train(self, overfit=False):
        logging.info(f"Starting training for {self.episodes} episodes.")
        for episode in range(1, self.episodes + 1):
            full_state = self.env.reset()
            # Split state into high and low
            high_state = full_state[:5]
            low_state = full_state[5:]
            total_reward = 0
            loss_high = 0.0
            loss_low = 0.0
            done = False

            while not done:
                # High-level agent selects action based on high_state
                high_action = self.high_agent.act(high_state)

                # Low-level agent selects action based on low_state (if applicable)
                low_action = self.low_agent.act(low_state)

                # Execute high-level action in the environment
                next_full_state, reward, done, _ = self.env.step(high_action)

                # Split next state
                next_high_state = next_full_state[:5]
                next_low_state = next_full_state[5:]

                # Store experience in high-level agent
                self.high_agent.remember(high_state, high_action, reward, next_high_state, done)

                # Optionally, store experience in low-level agent
                # Example: Modify reward or environment based on low_action if needed
                # For now, assuming low-level actions don't influence the environment
                # self.low_agent.remember(low_state, low_action, modified_reward, next_low_state, done)

                # Replay experiences and accumulate loss
                loss_high += self.high_agent.replay()
                # loss_low += self.low_agent.replay()  # Uncomment if low_agent has experiences to replay

                # Update states
                high_state = next_high_state
                low_state = next_low_state
                total_reward += reward

            # Update target networks periodically
            if episode % self.target_update_freq == 0:
                self.high_agent.update_target_model()
                self.low_agent.update_target_model()
                logging.info(f"Updated target networks at episode {episode}.")

            # Log episode statistics
            logging.info(f"Episode {episode}/{self.episodes} - Reward: {total_reward} - "
                         f"Loss High: {loss_high:.4f} - Loss Low: {loss_low:.4f} - "
                         f"Epsilon High: {self.high_agent.epsilon:.4f} - Epsilon Low: {self.low_agent.epsilon:.4f}")

            # Store loss history
            self.loss_high_history.append(loss_high)
            self.loss_low_history.append(loss_low)

        logging.info("Training completed.")

        # Plot training loss
        Visuals.plot_loss(self.loss_high_history, self.loss_low_history, save_path='results/training_loss.png')
