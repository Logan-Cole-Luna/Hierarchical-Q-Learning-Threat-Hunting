# agents/high_level_agent.py

from agents.base_agent import BaseAgent

class HighLevelAgent(BaseAgent):
    def __init__(self, state_size, action_size, hidden_layers, lr, gamma, epsilon, epsilon_min, epsilon_decay, memory_capacity, batch_size, device):
        super(HighLevelAgent, self).__init__(
            state_size, action_size, hidden_layers, lr, gamma, epsilon, epsilon_min, epsilon_decay, memory_capacity, batch_size, device
        )
        # Additional initialization for high-level agent if needed
