"""
Core game logic for the Tic-Tac-Toe game.
This module implements the game rules, board management, and win condition checking.
"""

import numpy as np

class TicTacToeGame:
    """
    Implements the core Tic-Tac-Toe game mechanics using a NumPy array as the board.
    The board uses the following representation:
    - 0: Empty cell
    - 1: Player X
    - -1: Player O
    """
    def __init__(self):
        # Initialize empty 3x3 board
        self.board = np.zeros((3, 3), dtype=int)  
        self.current_player = 1  # X starts (1 for X, -1 for O)
        self.winner = None       # None: ongoing, 0: tie, 1: X wins, -1: O wins
        self.game_over = False
    
    def make_move(self, row, col):
        """
        Attempt to make a move at the specified position.
        
        Args:
            row (int): Row index (0-2)
            col (int): Column index (0-2)
            
        Returns:
            bool: True if move was valid and executed, False otherwise
        """
        if self.is_valid_move(row, col):
            self.board[row, col] = self.current_player
            if self.check_winner():
                self.game_over = True
                self.winner = self.current_player
            elif self.is_board_full():
                self.game_over = True
                self.winner = 0  # Tie game
            else:
                self.current_player *= -1  # Switch players
            return True
        return False
    
    def is_valid_move(self, row, col):
        """
        Check if a move is valid (within bounds and cell is empty).
        
        Args:
            row (int): Row index
            col (int): Column index
            
        Returns:
            bool: True if move is valid, False otherwise
        """
        return (
            0 <= row < 3 and 
            0 <= col < 3 and 
            self.board[row, col] == 0 and 
            not self.game_over
        )
    
    def get_valid_moves(self):
        """
        Get list of all valid moves in current game state.
        
        Returns:
            list: List of (row, col) tuples representing valid moves
        """
        return [(r, c) for r in range(3) for c in range(3) 
                if self.board[r, c] == 0]
    
    def check_winner(self):
        """
        Check if the current player has won.
        Checks all rows, columns, and diagonals for three in a row.
        
        Returns:
            bool: True if current player has won, False otherwise
        """
        # Check rows and columns
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                return True
        
        # Check diagonals
        if abs(sum(np.diag(self.board))) == 3:
            return True
        if abs(sum(np.diag(np.fliplr(self.board)))) == 3:
            return True
        
        return False
    
    def is_board_full(self):
        """
        Check if the board is full (tie game).
        
        Returns:
            bool: True if no empty cells remain, False otherwise
        """
        return not any(0 in row for row in self.board)
    
    def get_state(self):
        """
        Get current board state as a flat array.
        
        Returns:
            numpy.array: Flattened board state
        """
        return self.board.flatten()
    
    def board_to_state(self):
        """
        Convert board array to string representation for Q-table key.
        
        Returns:
            str: String representation of board state
        """
        return ''.join(map(str, self.board.flatten()))
    
    def reset(self):
        """Reset the game to initial state."""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.winner = None
        self.game_over = False
    
    def render(self):
        """
        Create string representation of current board state.
        
        Returns:
            str: ASCII representation of the board
        """
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        board_str = ""
        for i in range(3):
            for j in range(3):
                board_str += symbols[self.board[i, j]]
                if j < 2:
                    board_str += ' | '
            if i < 2:
                board_str += '\n---------\n'
        return board_str