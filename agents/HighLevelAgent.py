# agents/HighLevelAgent.py

from .BaseAgent import BaseAgent

class HighLevelAgent(BaseAgent):
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64):
        super().__init__(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size
        )
        # Additional initialization for high-level agent if needed
