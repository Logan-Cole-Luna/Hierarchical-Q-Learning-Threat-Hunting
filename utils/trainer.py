# utils/trainer.py

import torch
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train(self, num_episodes):
        reward_history = []
        loss_history = []

        for episode in range(1, num_episodes + 1):
            state_batch, labels = self.env.reset()  # Initial batch of states and labels
            done = False
            total_reward = 0
            losses = []

            logger.info(f"Starting Episode {episode}/{num_episodes}")

            while not done:
                actions = []
                for state in state_batch:
                    action = self.agent.act(state)
                    actions.append(action)
                actions = np.array(actions)
                logger.debug(f"Actions for current batch: {actions}")

                try:
                    next_state_batch, rewards, done, info = self.env.step(actions)
                except ValueError as ve:
                    logger.error(f"ValueError during step: {ve}")
                    break

                logger.debug(f"Received next_state_batch shape: {next_state_batch.shape}, rewards shape: {rewards.shape}, done: {done}")

                # Iterate over each sample in the batch
                for state, action, reward, next_state in zip(state_batch, actions, rewards, next_state_batch):
                    loss = self.agent.learn(state, action, reward, next_state, done)
                    if loss is not None:
                        losses.append(loss)
                    total_reward += reward

                state_batch = next_state_batch

            reward_history.append(total_reward)
            if losses:
                loss_history.append(np.mean(losses))

            # Log progress every 10 episodes and the first episode
            if episode % 10 == 0 or episode == 1:
                avg_reward = np.mean(reward_history[-10:]) if episode >= 10 else np.mean(reward_history)
                avg_loss = np.mean(loss_history[-10:]) if len(loss_history) >= 10 else (np.mean(loss_history) if loss_history else 0)
                logger.info(f"Episode {episode}/{num_episodes} - Total Reward: {total_reward:.2f} - Average Reward (last 10): {avg_reward:.2f} - Average Loss: {avg_loss:.4f}")

        logger.info("Training completed.")
        return reward_history, loss_history
