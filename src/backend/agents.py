"""
Implementation of various AI agents for playing Tic-Tac-Toe.
This module contains different agent types ranging from simple random moves to perfect play.
"""

import numpy as np
import random
from .game import TicTacToeGame

class Agent:
    """Base class for all Tic-Tac-Toe agents."""
    def get_move(self, game):
        """
        Get the agent's next move for the current game state.
        Must be implemented by all agent classes.
        
        Args:
            game: TicTacToeGame instance representing current game state
            
        Returns:
            tuple: (row, col) representing the chosen move
        """
        raise NotImplementedError

class HumanAgent(Agent):
    """Human player that inputs moves via console interface."""
    def get_move(self, game):
        """
        Get move from human player through console input.
        Validates input and ensures move is legal.
        
        Args:
            game: TicTacToeGame instance
            
        Returns:
            tuple: (row, col) of chosen valid move
        """
        valid_moves = game.get_valid_moves()
        while True:
            try:
                row = int(input("Enter row (0-2): "))
                col = int(input("Enter column (0-2): "))
                if (row, col) in valid_moves:
                    return row, col
                print("Invalid move, try again.")
            except ValueError:
                print("Please enter numbers between 0 and 2.")

class RandomAgent(Agent):
    """Agent that makes random valid moves."""
    def get_move(self, game):
        """
        Choose a random valid move from available moves.
        
        Args:
            game: TicTacToeGame instance
            
        Returns:
            tuple: Random (row, col) from valid moves
        """
        valid_moves = game.get_valid_moves()
        return random.choice(valid_moves)

class MinimaxAgent(Agent):
    """
    Perfect player using minimax algorithm with alpha-beta pruning.
    Implements optimal strategy that cannot lose.
    """
    def __init__(self):
        self.cache = {}  # Cache previously computed positions for efficiency
    
    def get_move(self, game):
        """
        Get optimal move using minimax algorithm.
        
        Args:
            game: TicTacToeGame instance
            
        Returns:
            tuple: (row, col) of optimal move
        """
        _, move = self._minimax(game.board.copy(), game.current_player, float('-inf'), float('inf'))
        return move
    
    def _minimax(self, board, player, alpha, beta):
        """
        Minimax algorithm with alpha-beta pruning for optimal move selection.
        
        Args:
            board: NumPy array of current board state
            player: Current player (1 for X, -1 for O)
            alpha: Best value maximizer can guarantee
            beta: Best value minimizer can guarantee
            
        Returns:
            tuple: (value, move) where value is the position value and move is (row, col)
        """
        # Convert board to tuple for hashing in cache
        board_tuple = tuple(board.flatten())
        cache_key = (board_tuple, player)
        
        # Return cached result if available
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Check terminal states
        winner = self._check_winner(board)
        if winner is not None:
            result = (winner, None)
            self.cache[cache_key] = result
            return result
        
        # Get valid moves
        valid_moves = [(r, c) for r in range(3) for c in range(3) if board[r, c] == 0]
        if not valid_moves:  # Tie game
            result = (0, None)
            self.cache[cache_key] = result
            return result
        
        # Maximizing player
        if player == 1:
            best_value = float('-inf')
            best_move = None
            for move in valid_moves:
                new_board = board.copy()
                new_board[move] = player
                value, _ = self._minimax(new_board, -player, alpha, beta)
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break  # Beta cutoff
            result = (best_value, best_move)
            self.cache[cache_key] = result
            return result
        
        # Minimizing player
        else:
            best_value = float('inf')
            best_move = None
            for move in valid_moves:
                new_board = board.copy()
                new_board[move] = player
                value, _ = self._minimax(new_board, -player, alpha, beta)
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)
                if beta <= alpha:
                    break  # Alpha cutoff
            result = (best_value, best_move)
            self.cache[cache_key] = result
            return result
    
    def _check_winner(self, board):
        """
        Check if there is a winner in the current board state.
        
        Args:
            board: NumPy array of current board state
            
        Returns:
            int: 1 if X wins, -1 if O wins, 0 if tie, None if game is not over
        """
        # Check rows
        for row in board:
            if all(x == row[0] for x in row) and row[0] != 0:
                return row[0]
        
        # Check columns
        for col in range(3):
            if all(board[row][col] == board[0][col] for row in range(3)) and board[0][col] != 0:
                return board[0][col]
        
        # Check diagonals
        if all(board[i][i] == board[0][0] for i in range(3)) and board[0][0] != 0:
            return board[0][0]
        if all(board[i][2 - i] == board[0][2] for i in range(3)) and board[0][2] != 0:
            return board[0][2]
        
        # Check for tie
        if not any(0 in row for row in board):
            return 0
        
        # Game is not over
        return None

class QLearningAgent(Agent):
    """
    Q-Learning agent that learns to play Tic-Tac-Toe through experience.
    Uses an epsilon-greedy strategy for exploration vs exploitation.
    """
    def __init__(self, learning_rate, discount_factor, initial_epsilon, min_epsilon, epsilon_decay_rate):
        """
        Initialize Q-Learning agent with learning parameters.
        
        Args:
            learning_rate (float): Rate at which agent learns from new experiences (0-1)
            discount_factor (float): Weight given to future rewards (0-1)
            initial_epsilon (float): Initial exploration rate (0-1)
            min_epsilon (float): Minimum exploration rate
            epsilon_decay_rate (float): Rate at which exploration decreases
        """
        self.q_table = {}  # State-action value table
        self.lr = learning_rate
        self.gamma = discount_factor
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay_rate  # Rate at which exploration decreases
        self.epsilon = initial_epsilon
        self.current_state = None
        self.current_move = None
        self.games_played = 0

    def reset(self):
        """
        Reset agent's episode memory and update exploration rate.
        Called at the end of each game.
        """
        self.current_state = None
        self.current_move = None
        self.games_played += 1
        # Decay epsilon for less exploration over time
        self.epsilon = max(self.min_epsilon,
                         self.initial_epsilon * (self.epsilon_decay ** self.games_played))

    def board_to_state(self, board):
        """
        Convert board array to string representation for Q-table key.
        
        Args:
            board: NumPy array of current board state
            
        Returns:
            str: String representation of board state
        """
        return ''.join(map(str, board.flatten()))

    def get_move(self, game):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            game: TicTacToeGame instance
            
        Returns:
            tuple: (row, col) of chosen move
        """
        self.current_state = self.board_to_state(game.board)
        valid_moves = game.get_valid_moves()
        epsilon_val = self.epsilon
        
        # Exploration vs exploitation
        if random.random() < epsilon_val:
            # Explore
            self.current_move = random.choice(valid_moves)
            return self.current_move
        else:
            # Exploit
            if self.current_state not in self.q_table:
                # Initialize Q-values for new state
                self.q_table[self.current_state] = {str(move): 0.0 for move in valid_moves}
            
            # Get Q-values for current state
            q_values = self.q_table[self.current_state]
            
            # Choose action with highest Q-value
            best_move = max(valid_moves, key=lambda move: q_values.get(str(move), 0.0))
            self.current_move = best_move
            return best_move

    def calculate_reward(self, game, next_state):
        """
        Calculate reward based on game outcome and strategic moves.
        
        Args:
            game: TicTacToeGame instance
            next_state: Next game state after move
            
        Returns:
            float: Reward value
        """
        if game.winner == game.current_player:
            return 5.0  # Win
        elif game.winner == 0:
            return 1.0  # Tie
        elif game.winner == -game.current_player:
            return -3.0  # Loss
        
        # Check if opponent is about to win and we're blocking
        opponent = -game.current_player
        blocked_win = False
        for move in game.get_valid_moves():
            game_copy = TicTacToeGame()
            game_copy.board = game.board.copy()
            game_copy.current_player = opponent
            game_copy.make_move(*move)
            if game_copy.winner == opponent:
                blocked_win = True
                break
            game_copy.board[move] = 0
        
        if blocked_win:
            return 2.0  # Reward for blocking opponent win
        
        # Check if we're setting up a potential win
        current_player = game.current_player
        potential_win = False
        for move in game.get_valid_moves():
            game_copy.board = game.board.copy()
            game_copy.board[move] = current_player
            if abs(sum(game_copy.board[i, :])) == 3 or \
               abs(sum(game_copy.board[:, i])) == 3 or \
               abs(sum(np.diag(game_copy.board))) == 3 or \
               abs(sum(np.diag(np.fliplr(game_copy.board)))) == 3:
                potential_win = True
                break
        
        if potential_win:
            return 1.5  # Reward for creating winning opportunity
        
        # Strategic position rewards
        if self.current_move == (1, 1):
            return 0.8  # Center position
        
        if self.current_move in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            return 0.5  # Corner positions
        
        # Penalize missing strategic positions early game
        if len(game.get_valid_moves()) > 4:
            if (1, 1) in game.get_valid_moves():
                return -0.2  # Missed center
            elif any(move in game.get_valid_moves() for move in [(0, 0), (0, 2), (2, 0), (2, 2)]):
                return -0.1  # Missed corner
        
        return 0.1  # Small positive reward for other moves

    def learn(self, state, action, reward, next_state, next_valid_moves):
        """
        Update Q-values using Q-learning update rule:
        Q(s,a) = Q(s,a) + lr * (reward + gamma * max(Q(s')) - Q(s,a))
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            next_valid_moves: Valid moves in next state
        """
        state_key = state
        action_key = str(action)
        
        # Initialize Q-values if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
            
        # Calculate max Q-value for next state
        if next_state is None or not next_valid_moves:  # Terminal state
            max_next_q = 0
        else:
            if next_state not in self.q_table:
                self.q_table[next_state] = {str(move): 0.0 for move in next_valid_moves}
            max_next_q = max(self.q_table[next_state].values())
        
        # Q-learning update rule
        old_q = self.q_table[state_key][action_key]
        new_q = old_q + self.lr * (reward + self.gamma * max_next_q - old_q)
        self.q_table[state_key][action_key] = new_q

    def get_q_values(self, state):
        """
        Get Q-values for all actions in a state.
        
        Args:
            state: Game state
            
        Returns:
            dict: Dictionary mapping actions to Q-values
        """
        if state not in self.q_table:
            return {}
        return self.q_table[state] 