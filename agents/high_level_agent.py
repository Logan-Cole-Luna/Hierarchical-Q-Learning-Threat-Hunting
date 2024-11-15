"""
high_level_agent.py

Defines the HighLevelAgent class, responsible for high-level actions in the hierarchical RL setup.
Inherits from BaseAgent and manages hierarchical coordination and target network updates.

Classes:
    - HighLevelAgent: Agent responsible for high-level category predictions.
"""

from agents.base_agent import BaseAgent

class HighLevelAgent(BaseAgent):
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64, target_update_freq=10):
        """
        Initializes the HighLevelAgent with additional parameters for target network updates.

        Parameters:
        -----------
        All parameters are inherited from BaseAgent, with an additional:
        target_update_freq : int, optional
            Frequency (in episodes) to update the target network (default is 10).
        """
        super().__init__(state_size, action_size, learning_rate, gamma, 
                         epsilon, epsilon_min, epsilon_decay, batch_size)
        self.target_update_freq = target_update_freq
        self.episode_count = 0

    def replay(self):
        """
        Extends the replay method to include periodic target network updates.
        """
        super().replay()
        self.episode_count += 1
        if self.episode_count % self.target_update_freq == 0:
            self.update_target_network()
            print(f"HighLevelAgent: Target network updated at episode {self.episode_count}")
