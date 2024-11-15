"""
low_level_agent.py

Defines the LowLevelAgent class, responsible for low-level actions in the hierarchical RL setup.
Inherits from BaseAgent and manages hierarchical coordination and target network updates.

Classes:
    - LowLevelAgent: Agent responsible for low-level anomaly predictions.
"""

from agents.base_agent import BaseAgent

class LowLevelAgent(BaseAgent):
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64, target_update_freq=10):
        """
        Initializes the LowLevelAgent with additional parameters for target network updates.

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

    def set_goal(self, high_level_action):
        """
        Adjusts the low-level agent's behavior based on the high-level agent's action.
        For example, focusing on specific anomaly types.

        Parameters:
        -----------
        high_level_action : int
            The high-level action selected by the high-level agent.
        """
        # Example implementation: Modify action space or priorities based on high-level action
        # This is application-specific and may vary based on your hierarchical setup
        self.current_goal = high_level_action
        # Implement any necessary adjustments here

    def replay(self):
        """
        Extends the replay method to include periodic target network updates.
        """
        super().replay()
        self.episode_count += 1
        if self.episode_count % self.target_update_freq == 0:
            self.update_target_network()
            print(f"LowLevelAgent: Target network updated at episode {self.episode_count}")
