# Trainer.py

from IntrusionDetectionEnv import IntrusionDetectionEnv
from agents.HighLevelAgent import HighLevelAgent
from agents.LowLevelAgent import LowLevelAgent
from RewardCalculator import RewardCalculator
import numpy as np

class Trainer:
    def __init__(self, env, high_agent, low_agent, episodes=1000):
        self.env = env
        self.high_agent = high_agent
        self.low_agent = low_agent
        self.episodes = episodes
    
    def train(self):
        for episode in range(1, self.episodes + 1):
            state_high, state_low, label = self.env.reset()
            state_high = np.reshape(state_high, [1, self.high_agent.state_size])
            state_low = np.reshape(state_low, [1, self.low_agent.state_size])
            done = False
            total_reward_high = 0
            total_reward_low = 0
            while not done:
                # Agents act
                action_high = self.high_agent.act(state_high)
                action_low = self.low_agent.act(state_low)
                
                # Environment step
                (next_state_high, next_state_low, next_label), (reward_high, reward_low), done = self.env.step(action_high, action_low)
                
                # Reshape next states
                if not done:
                    next_state_high = np.reshape(next_state_high, [1, self.high_agent.state_size])
                    next_state_low = np.reshape(next_state_low, [1, self.low_agent.state_size])
                
                # Remember experiences
                self.high_agent.remember(state_high, action_high, reward_high, next_state_high, done)
                self.low_agent.remember(state_low, action_low, reward_low, next_state_low, done)
                
                # Update states
                state_high = next_state_high
                state_low = next_state_low
                
                # Accumulate rewards
                total_reward_high += reward_high
                total_reward_low += reward_low
            
            # Replay agents
            self.high_agent.replay()
            self.low_agent.replay()
            
            # Print progress every 100 episodes
            if episode % 100 == 0 or episode == 1:
                print(f"Episode {episode}/{self.episodes} - High Reward: {total_reward_high}, Low Reward: {total_reward_low}")
