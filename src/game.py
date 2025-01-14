import numpy as np

class TicTacToeGame:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # 0 for empty, 1 for X, -1 for O
        self.current_player = 1  # X starts
        self.winner = None
        self.game_over = False
    
    def make_move(self, row, col):
        """Make a move on the board. Returns True if move was valid."""
        if self.is_valid_move(row, col):
            self.board[row, col] = self.current_player
            if self.check_winner():
                self.game_over = True
                self.winner = self.current_player
            elif self.is_board_full():
                self.game_over = True
                self.winner = 0  # Tie
            else:
                self.current_player *= -1  # Switch player
            return True
        return False
    
    def is_valid_move(self, row, col):
        """Check if a move is valid."""
        return (
            0 <= row < 3 and 
            0 <= col < 3 and 
            self.board[row, col] == 0 and 
            not self.game_over
        )
    
    def get_valid_moves(self):
        """Return list of valid moves as (row, col) tuples."""
        return [(r, c) for r in range(3) for c in range(3) 
                if self.board[r, c] == 0]
    
    def check_winner(self):
        """Check if current player has won."""
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
        """Check if the board is full (tie)."""
        return not any(0 in row for row in self.board)
    
    def get_state(self):
        """Return current state of the board as a flat array."""
        return self.board.flatten()
    
    def reset(self):
        """Reset the game to initial state."""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.winner = None
        self.game_over = False
    
    def render(self):
        """Return string representation of the board."""
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