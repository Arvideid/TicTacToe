from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from backend.game import TicTacToeGame
from backend.agents import RandomAgent, MinimaxAgent, QLearningAgent, HumanAgent
import os
import traceback

app = Flask(__name__, 
    static_url_path='',
    static_folder='static',
    template_folder='templates'
)
CORS(app)  # Enable CORS for all routes

# Initialize game and agents
game = TicTacToeGame()
agents = {
    "random": RandomAgent(),
    "perfect": MinimaxAgent(),
    "learning": QLearningAgent(
        learning_rate=0.4,
        discount_factor=0.95,
        initial_epsilon=0.9,
        min_epsilon=0.05,
        epsilon_decay_rate=0.997
    ),
}

# Track performance stats
performance_stats = {
    'vs_random': {'wins': [], 'ties': [], 'losses': []},
    'vs_perfect': {'wins': [], 'ties': [], 'losses': []}
}
episodes = []
games_played = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/move', methods=['POST'])
def make_move():
    try:
        global games_played
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        row = data.get('row')
        col = data.get('col')
        agent_type = data.get('agent', '').lower()  # Convert to lowercase
        
        # Make human move if coordinates provided
        if row is not None and col is not None:
            if not game.is_valid_move(row, col):
                return jsonify({'error': 'Invalid move position'}), 400
            game.make_move(row, col)
        
        # Make AI move if game is not over
        if not game.game_over and agent_type in agents:
            ai_agent = agents[agent_type]
            ai_row, ai_col = ai_agent.get_move(game)
            if game.is_valid_move(ai_row, ai_col):
                game.make_move(ai_row, ai_col)
            else:
                return jsonify({'error': 'Invalid AI move'}), 500

        # Update performance stats if game is over
        if game.game_over:
            games_played += 1
            episodes.append(games_played)
            
            # Update stats for both vs_random and vs_perfect
            for opponent in ['random', 'perfect']:
                if games_played > len(performance_stats[f'vs_{opponent}']['wins']):
                    if game.winner == 1:  # Learning AI won
                        performance_stats[f'vs_{opponent}']['wins'].append(
                            len(performance_stats[f'vs_{opponent}']['wins']) + 1)
                        performance_stats[f'vs_{opponent}']['ties'].append(
                            len(performance_stats[f'vs_{opponent}']['ties']))
                        performance_stats[f'vs_{opponent}']['losses'].append(
                            len(performance_stats[f'vs_{opponent}']['losses']))
                    elif game.winner == -1:  # Learning AI lost
                        performance_stats[f'vs_{opponent}']['wins'].append(
                            len(performance_stats[f'vs_{opponent}']['wins']))
                        performance_stats[f'vs_{opponent}']['ties'].append(
                            len(performance_stats[f'vs_{opponent}']['ties']))
                        performance_stats[f'vs_{opponent}']['losses'].append(
                            len(performance_stats[f'vs_{opponent}']['losses']) + 1)
                    else:  # Tie
                        performance_stats[f'vs_{opponent}']['wins'].append(
                            len(performance_stats[f'vs_{opponent}']['wins']))
                        performance_stats[f'vs_{opponent}']['ties'].append(
                            len(performance_stats[f'vs_{opponent}']['ties']) + 1)
                        performance_stats[f'vs_{opponent}']['losses'].append(
                            len(performance_stats[f'vs_{opponent}']['losses']))
        
        return jsonify({
            'board': game.board.tolist(),
            'current_player': 'X' if game.current_player == 1 else 'O',
            'game_over': game.game_over,
            'winner': game.winner,
            'episodes': episodes,
            'stats': performance_stats
        })
    except Exception as e:
        print(f"Error in make_move: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_game():
    try:
        game.reset()
        return jsonify({
            'board': game.board.tolist(),
            'current_player': 'X' if game.current_player == 1 else 'O',
            'game_over': game.game_over,
            'winner': game.winner
        })
    except Exception as e:
        print(f"Error in reset_game: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 