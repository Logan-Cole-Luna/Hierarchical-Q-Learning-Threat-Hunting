# agents/low_level_agent.py

from agents.base_agent import BaseAgent

class LowLevelAgent(BaseAgent):
    def __init__(self, state_size, action_size, hidden_layers, lr, gamma, epsilon, epsilon_min, epsilon_decay, memory_capacity, batch_size, device):
        super(LowLevelAgent, self).__init__(
            state_size, action_size, hidden_layers, lr, gamma, epsilon, epsilon_min, epsilon_decay, memory_capacity, batch_size, device
        )
        # Additional initialization for low-level agent if needed
