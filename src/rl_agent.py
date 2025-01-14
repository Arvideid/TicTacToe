import numpy as np
import json
import os
from agents import Agent

class QLearningAgent(Agent):
    def __init__(self, learning_rate=0.3, discount_factor=0.9, epsilon=0.2):
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.current_state = None
        self.current_move = None
        self.load_q_table()

    def reset(self):
        """Reset the agent's episode memory."""
        self.current_state = None
        self.current_move = None

    def board_to_state(self, board):
        """Convert board to a string state representation."""
        return ''.join(map(str, board.flatten()))

    def get_move(self, game):
        """Choose action using epsilon-greedy policy."""
        state = self.board_to_state(game.board)
        valid_moves = game.get_valid_moves()
        
        # Initialize state if not seen before
        if state not in self.q_table:
            self.q_table[state] = {str(move): 0.0 for move in valid_moves}
        
        # Store current state
        self.current_state = state
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Random move
            move = valid_moves[np.random.randint(len(valid_moves))]
        else:
            # Get Q-values for valid moves
            q_values = {move: self.q_table[state].get(str(move), 0.0) 
                       for move in valid_moves}
            
            # Get moves with maximum Q-value
            max_q = max(q_values.values())
            best_moves = [move for move, q in q_values.items() 
                         if q == max_q]
            
            move = best_moves[np.random.randint(len(best_moves))]
        
        self.current_move = move
        return move

    def learn(self, state, action, reward, next_state, next_valid_moves):
        """Update Q-values using Q-learning update rule."""
        # Convert state and action to proper format
        state_key = state
        action_key = str(action)
        
        # Initialize Q-values if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
            
        # Get max Q-value for next state
        if next_state is None or not next_valid_moves:  # Terminal state
            max_next_q = 0
        else:
            if next_state not in self.q_table:
                self.q_table[next_state] = {str(move): 0.0 for move in next_valid_moves}
            max_next_q = max(self.q_table[next_state].values())
        
        # Update Q-value
        old_q = self.q_table[state_key][action_key]
        new_q = old_q + self.lr * (reward + self.gamma * max_next_q - old_q)
        self.q_table[state_key][action_key] = new_q
        
        # Save after each update
        self.save_q_table()

    def get_q_values(self, state):
        """Get Q-values for all actions in a state."""
        if state not in self.q_table:
            return {}
        return self.q_table[state]

    def save_q_table(self):
        """Save Q-table to a file."""
        try:
            with open('q_table.json', 'w') as f:
                json.dump(self.q_table, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save Q-table: {e}")

    def load_q_table(self):
        """Load Q-table from file if it exists."""
        try:
            with open('q_table.json', 'r') as f:
                self.q_table = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.q_table = {} 