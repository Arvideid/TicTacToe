import json
import os
from datetime import datetime

class TrainingStats:
    def __init__(self):
        # Game statistics
        self.games_played = 0
        self.x_wins = 0
        self.o_wins = 0
        self.ties = 0
        
        # Training statistics
        self.training_history = {
            'episodes': [],
            'vs_random': {
                'win_rate': [],
                'tie_rate': [],
                'loss_rate': []
            },
            'vs_minimax': {
                'win_rate': [],
                'tie_rate': [],
                'loss_rate': []
            },
            'timestamps': []
        }
        
        # Load existing stats if available
        self.load_stats()
    
    def add_evaluation_result(self, episode, opponent_type, wins, losses, ties, total_games):
        """Add a new evaluation result to the training history."""
        win_rate = wins / total_games
        tie_rate = ties / total_games
        loss_rate = losses / total_games
        
        self.training_history['episodes'].append(episode)
        
        if opponent_type == 'random':
            self.training_history['vs_random']['win_rate'].append(win_rate)
            self.training_history['vs_random']['tie_rate'].append(tie_rate)
            self.training_history['vs_random']['loss_rate'].append(loss_rate)
        elif opponent_type == 'minimax':
            self.training_history['vs_minimax']['win_rate'].append(win_rate)
            self.training_history['vs_minimax']['tie_rate'].append(tie_rate)
            self.training_history['vs_minimax']['loss_rate'].append(loss_rate)
        
        self.training_history['timestamps'].append(datetime.now().isoformat())
        self.save_stats()
    
    def get_training_stats(self):
        """Get the current training statistics."""
        if not self.training_history['episodes']:
            return None
        return self.training_history
    
    def save_stats(self):
        """Save statistics to a JSON file."""
        try:
            stats_data = {
                'game_stats': {
                    'games_played': self.games_played,
                    'x_wins': self.x_wins,
                    'o_wins': self.o_wins,
                    'ties': self.ties
                },
                'training_history': self.training_history
            }
            
            with open('training_stats.json', 'w') as f:
                json.dump(stats_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save training stats: {e}")
    
    def load_stats(self):
        """Load statistics from JSON file if it exists."""
        try:
            if os.path.exists('training_stats.json'):
                with open('training_stats.json', 'r') as f:
                    stats_data = json.load(f)
                    
                    # Load game stats
                    game_stats = stats_data.get('game_stats', {})
                    self.games_played = game_stats.get('games_played', 0)
                    self.x_wins = game_stats.get('x_wins', 0)
                    self.o_wins = game_stats.get('o_wins', 0)
                    self.ties = game_stats.get('ties', 0)
                    
                    # Load training history
                    self.training_history = stats_data.get('training_history', self.training_history)
        except Exception as e:
            print(f"Warning: Could not load training stats: {e}")
    
    def reset_game_stats(self):
        """Reset game statistics."""
        self.games_played = 0
        self.x_wins = 0
        self.o_wins = 0
        self.ties = 0
        self.save_stats()
    
    def reset_training_stats(self):
        """Reset training statistics."""
        self.training_history = {
            'episodes': [],
            'vs_random': {
                'win_rate': [],
                'tie_rate': [],
                'loss_rate': []
            },
            'vs_minimax': {
                'win_rate': [],
                'tie_rate': [],
                'loss_rate': []
            },
            'timestamps': []
        }
        self.save_stats() 