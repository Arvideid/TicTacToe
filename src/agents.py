import numpy as np
import random

class Agent:
    """Base class for all agents."""
    def get_move(self, game):
        raise NotImplementedError

class HumanAgent(Agent):
    """Human player that inputs moves via console."""
    def get_move(self, game):
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
        valid_moves = game.get_valid_moves()
        return random.choice(valid_moves)

class MinimaxAgent(Agent):
    """Perfect player using minimax algorithm with alpha-beta pruning."""
    def __init__(self):
        self.cache = {}  # Store previously computed positions
    
    def get_move(self, game):
        _, move = self._minimax(game.board.copy(), game.current_player, float('-inf'), float('inf'))
        return move
    
    def _minimax(self, board, player, alpha, beta):
        """
        Minimax algorithm with alpha-beta pruning.
        board: numpy array of the current board state
        player: 1 for X, -1 for O
        alpha: best value that maximizer can guarantee
        beta: best value that minimizer can guarantee
        """
        # Convert board to tuple for hashing
        board_tuple = tuple(board.flatten())
        cache_key = (board_tuple, player)
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Check for terminal states
        winner = self._check_winner(board)
        if winner is not None:
            result = (winner, None)
            self.cache[cache_key] = result
            return result
        
        valid_moves = [(r, c) for r in range(3) for c in range(3) if board[r, c] == 0]
        if not valid_moves:  # Tie
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
                if beta <= alpha:
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
                if beta <= alpha:
                    break
        
        result = (best_value, best_move)
        self.cache[cache_key] = result
        return result
    
    def _check_winner(self, board):
        """Check if there's a winner on the board."""
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