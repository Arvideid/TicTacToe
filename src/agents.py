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
    """Perfect player using minimax algorithm."""
    def get_move(self, game):
        _, move = self._minimax(game, True)
        return move

    def _minimax(self, game, is_maximizing):
        if game.game_over:
            if game.winner == 1:  # X wins
                return 1, None
            elif game.winner == -1:  # O wins
                return -1, None
            else:  # Tie
                return 0, None

        valid_moves = game.get_valid_moves()
        best_move = valid_moves[0]
        
        if is_maximizing:
            best_value = float('-inf')
            for move in valid_moves:
                # Create a copy of the game state
                game_copy = game.__class__()
                game_copy.board = game.board.copy()
                game_copy.current_player = game.current_player
                
                game_copy.make_move(*move)
                value, _ = self._minimax(game_copy, False)
                
                if value > best_value:
                    best_value = value
                    best_move = move
        else:
            best_value = float('inf')
            for move in valid_moves:
                # Create a copy of the game state
                game_copy = game.__class__()
                game_copy.board = game.board.copy()
                game_copy.current_player = game.current_player
                
                game_copy.make_move(*move)
                value, _ = self._minimax(game_copy, True)
                
                if value < best_value:
                    best_value = value
                    best_move = move
                    
        return best_value, best_move 