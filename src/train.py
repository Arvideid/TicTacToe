import numpy as np
import matplotlib.pyplot as plt
from game import TicTacToeGame
from agents import RandomAgent, MinimaxAgent
from rl_agent import QLearningAgent

def train_agent(episodes=10000, eval_interval=500):
    # Initialize agents and game
    rl_agent = QLearningAgent()
    random_opponent = RandomAgent()
    minimax_opponent = MinimaxAgent()
    game = TicTacToeGame()
    
    # Training metrics
    random_winrates = []
    minimax_winrates = []
    episodes_x = []
    
    # Training loop
    for episode in range(episodes):
        game.reset()
        rl_agent.reset()
        
        # Training game against random opponent
        while not game.game_over:
            # RL agent's turn
            if game.current_player == 1:
                move = rl_agent.get_move(game)
                game.make_move(*move)
                
                # Learn from the move
                if game.game_over:
                    reward = 1.0 if game.winner == 1 else (-1.0 if game.winner == -1 else 0.0)
                    rl_agent.learn(reward, game)
                else:
                    rl_agent.learn(0.0, game)
            
            # Random opponent's turn
            else:
                move = random_opponent.get_move(game)
                game.make_move(*move)
        
        # Evaluation phase
        if (episode + 1) % eval_interval == 0:
            random_wins = 0
            minimax_wins = 0
            n_eval_games = 100
            
            # Evaluate against random opponent
            for _ in range(n_eval_games):
                game.reset()
                while not game.game_over:
                    if game.current_player == 1:
                        move = rl_agent.get_move(game)
                    else:
                        move = random_opponent.get_move(game)
                    game.make_move(*move)
                if game.winner == 1:
                    random_wins += 1
            
            # Evaluate against minimax opponent
            for _ in range(n_eval_games):
                game.reset()
                while not game.game_over:
                    if game.current_player == 1:
                        move = rl_agent.get_move(game)
                    else:
                        move = minimax_opponent.get_move(game)
                    game.make_move(*move)
                if game.winner == 1:
                    minimax_wins += 1
            
            # Record metrics
            random_winrates.append(random_wins / n_eval_games)
            minimax_winrates.append(minimax_wins / n_eval_games)
            episodes_x.append(episode + 1)
            
            print(f"Episode {episode + 1}")
            print(f"Winrate vs Random: {random_wins/n_eval_games:.2%}")
            print(f"Winrate vs Minimax: {minimax_wins/n_eval_games:.2%}")
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_x, random_winrates, label='vs Random')
    plt.plot(episodes_x, minimax_winrates, label='vs Minimax')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate')
    plt.title('RL Agent Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.close()
    
    return rl_agent

if __name__ == "__main__":
    trained_agent = train_agent() 