import numpy as np
from agents import Agent

class QLearningAgent(Agent):
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = {}  # State-action value table
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.last_state = None
        self.last_action = None

    def _get_state_key(self, state):
        """Convert numpy array state to hashable tuple."""
        return tuple(map(int, state))  # Convert numpy values to regular integers

    def get_move(self, game):
        """Choose action using epsilon-greedy policy."""
        state = self._get_state_key(game.get_state())
        valid_moves = game.get_valid_moves()

        # Initialize Q-values for new state
        if state not in self.q_table:
            self.q_table[state] = {move: 0.0 for move in valid_moves}

        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action = valid_moves[np.random.randint(len(valid_moves))]
        else:
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = best_actions[np.random.randint(len(best_actions))]

        self.last_state = state
        self.last_action = action
        return action

    def learn(self, reward, new_game):
        """Update Q-values using Q-learning update rule."""
        if self.last_state is None:
            return

        new_state = self._get_state_key(new_game.get_state())

        # Initialize Q-values for new state if needed
        if new_state not in self.q_table:
            self.q_table[new_state] = {move: 0.0 for move in new_game.get_valid_moves()}

        # Get max Q-value for next state
        if new_game.game_over:
            max_next_q = 0
        else:
            max_next_q = max(self.q_table[new_state].values())

        # Update Q-value
        current_q = self.q_table[self.last_state][self.last_action]
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[self.last_state][self.last_action] = new_q

    def reset(self):
        """Reset the agent's episode memory."""
        self.last_state = None
        self.last_action = None 