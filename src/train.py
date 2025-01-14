import numpy as np
import matplotlib.pyplot as plt
from game import TicTacToeGame
from agents import RandomAgent, MinimaxAgent
from rl_agent import QLearningAgent
from training_stats import TrainingStats
import time

def train_agent(episodes=10000, eval_interval=100, eval_games=100, progress_callback=None):
    """Train the Q-learning agent and evaluate its performance."""
    # Initialize components
    rl_agent = QLearningAgent()
    random_opponent = RandomAgent()
    minimax_opponent = MinimaxAgent()
    game = TicTacToeGame()
    stats = TrainingStats()
    
    print("Starting training...")
    start_time = time.time()
    
    # Training loop
    for episode in range(episodes):
        game.reset()
        rl_agent.reset()
        
        # Update progress
        if progress_callback:
            progress_callback(episode)
        
        # Training game against random opponent
        while not game.game_over:
            # RL agent's turn
            if game.current_player == 1:
                state = rl_agent.board_to_state(game.board)
                move = rl_agent.get_move(game)
                game.make_move(*move)
                
                # Learn from the move
                next_state = rl_agent.board_to_state(game.board) if not game.game_over else None
                next_moves = game.get_valid_moves() if not game.game_over else []
                
                if game.game_over:
                    reward = 1.0 if game.winner == 1 else (-1.0 if game.winner == -1 else 0.1)
                else:
                    reward = 0.0
                
                rl_agent.learn(state, move, reward, next_state, next_moves)
            
            # Random opponent's turn
            else:
                move = random_opponent.get_move(game)
                game.make_move(*move)
        
        # Evaluation phase
        if (episode + 1) % eval_interval == 0:
            print(f"\nEvaluation at episode {episode + 1}")
            
            # Evaluate against random opponent
            random_results = evaluate_agent(rl_agent, random_opponent, eval_games)
            stats.add_evaluation_result(
                episode + 1, 
                'random',
                random_results['wins'],
                random_results['losses'],
                random_results['ties'],
                eval_games
            )
            print(f"vs Random - Win Rate: {random_results['wins']/eval_games:.2%}")
            
            # Evaluate against minimax opponent
            minimax_results = evaluate_agent(rl_agent, minimax_opponent, eval_games)
            stats.add_evaluation_result(
                episode + 1,
                'minimax',
                minimax_results['wins'],
                minimax_results['losses'],
                minimax_results['ties'],
                eval_games
            )
            print(f"vs Minimax - Win Rate: {minimax_results['wins']/eval_games:.2%}")
            
            # Save training progress plot
            stats.plot_training_progress('training_progress.png')
    
    # Final progress update
    if progress_callback:
        progress_callback(episodes)
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.1f} seconds")
    print(f"Final evaluation results:")
    print(f"vs Random - Win Rate: {random_results['wins']/eval_games:.2%}")
    print(f"vs Minimax - Win Rate: {minimax_results['wins']/eval_games:.2%}")
    
    return rl_agent

def evaluate_agent(agent, opponent, n_games):
    """Evaluate an agent against an opponent."""
    results = {'wins': 0, 'losses': 0, 'ties': 0}
    game = TicTacToeGame()
    
    for _ in range(n_games):
        game.reset()
        while not game.game_over:
            # Agent's turn
            if game.current_player == 1:
                move = agent.get_move(game)
            # Opponent's turn
            else:
                move = opponent.get_move(game)
            game.make_move(*move)
        
        if game.winner == 1:
            results['wins'] += 1
        elif game.winner == -1:
            results['losses'] += 1
        else:
            results['ties'] += 1
    
    return results

if __name__ == "__main__":
    # Train with different parameters for experimentation
    TRAINING_PARAMS = [
        {'episodes': 1000, 'eval_interval': 50, 'eval_games': 100},
        {'episodes': 5000, 'eval_interval': 100, 'eval_games': 100},
        {'episodes': 10000, 'eval_interval': 200, 'eval_games': 100}
    ]
    
    print("Select training configuration:")
    for i, params in enumerate(TRAINING_PARAMS):
        print(f"{i+1}. Episodes: {params['episodes']}, "
              f"Eval Interval: {params['eval_interval']}, "
              f"Eval Games: {params['eval_games']}")
    
    choice = int(input("Enter choice (1-3): ")) - 1
    trained_agent = train_agent(**TRAINING_PARAMS[choice]) 