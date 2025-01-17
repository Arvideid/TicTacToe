"""
Implementation of various AI agents for playing Tic-Tac-Toe.
This module contains different agent types ranging from simple random moves to perfect play.
"""

import numpy as np
import random

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
        
        best_move = valid_moves[0]
        if player == 1:  # Maximizing player
            best_value = float('-inf')
            for move in valid_moves:
                board_copy = board.copy()
                board_copy[move] = player
                value, _ = self._minimax(board_copy, -player, alpha, beta)
                
                if value > best_value:
                    best_value = value
                    best_move = move
                
                alpha = max(alpha, best_value)
                if beta <= alpha:  # Beta cutoff
                    break
        else:  # Minimizing player
            best_value = float('inf')
            for move in valid_moves:
                board_copy = board.copy()
                board_copy[move] = player
                value, _ = self._minimax(board_copy, -player, alpha, beta)
                
                if value < best_value:
                    best_value = value
                    best_move = move
                
                beta = min(beta, best_value)
                if beta <= alpha:  # Alpha cutoff
                    break
        
        result = (best_value, best_move)
        self.cache[cache_key] = result
        return result
    
    def _check_winner(self, board):
        """
        Check if there's a winner on the board.
        
        Args:
            board: NumPy array of current board state
            
        Returns:
            int or None: 1 for X win, -1 for O win, None if no winner
        """
        # Check rows and columns
        for i in range(3):
            if abs(sum(board[i, :])) == 3:
                return board[i, 0]
            if abs(sum(board[:, i])) == 3:
                return board[0, i]
        
        # Check diagonals
        diag_sum = sum(board[i, i] for i in range(3))
        if abs(diag_sum) == 3:
            return board[0, 0]
        
        anti_diag_sum = sum(board[i, 2-i] for i in range(3))
        if abs(anti_diag_sum) == 3:
            return board[0, 2]
        
        return None

class QLearningAgent(Agent):
    """
    Q-Learning agent that learns to play Tic-Tac-Toe through experience.
    Uses an epsilon-greedy strategy for exploration vs exploitation.
    """
    def __init__(self, learning_rate=0.4, discount_factor=0.95, initial_epsilon=0.9):
        """
        Initialize Q-Learning agent with learning parameters.
        
        Args:
            learning_rate (float): Rate at which agent learns from new experiences (0-1)
            discount_factor (float): Weight given to future rewards (0-1)
            initial_epsilon (float): Initial exploration rate (0-1)
        """
        self.q_table = {}  # State-action value table
        self.lr = learning_rate
        self.gamma = discount_factor
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = 0.05  # Minimum exploration rate
        self.epsilon_decay = 0.997  # Rate at which exploration decreases
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
        state = self.board_to_state(game.board)
        valid_moves = game.get_valid_moves()
        
        # Initialize Q-values for new state
        if state not in self.q_table:
            self.q_table[state] = {str(move): 0.0 for move in valid_moves}
        
        self.current_state = state
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Exploration: choose random move
            move = valid_moves[np.random.randint(len(valid_moves))]
        else:
            # Exploitation: choose best known move
            q_values = {move: self.q_table[state].get(str(move), 0.0) 
                       for move in valid_moves}
            
            # Get moves with maximum Q-value (handle ties randomly)
            max_q = max(q_values.values())
            best_moves = [move for move, q in q_values.items() 
                         if q == max_q]
            
            move = best_moves[np.random.randint(len(best_moves))]
        
        self.current_move = move
        return move

    def calculate_reward(self, game, next_state):
        """
        Calculate reward for the last action taken.
        Implements a sophisticated reward system considering:
        - Game outcomes (win/loss/tie)
        - Strategic positions (center, corners)
        - Blocking opponent wins
        - Setting up winning opportunities
        
        Args:
            game: TicTacToeGame instance
            next_state: State after action
            
        Returns:
            float: Calculated reward value
        """
        if game.game_over:
            if game.winner == game.current_player:
                return 5.0  # Win
            elif game.winner == 0:
                return 1.0  # Tie
            else:
                return -3.0  # Loss
        
        # Check if we blocked opponent's win
        opponent = -game.current_player
        game_copy = game.__class__()
        game_copy.board = game.board.copy()
        game_copy.current_player = opponent
        
        # Try opponent moves to detect blocked wins
        blocked_win = False
        for move in game_copy.get_valid_moves():
            game_copy.board[move] = opponent
            if abs(sum(game_copy.board[i, :])) == 3 or \
               abs(sum(game_copy.board[:, i])) == 3 or \
               abs(sum(np.diag(game_copy.board))) == 3 or \
               abs(sum(np.diag(np.fliplr(game_copy.board)))) == 3:
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