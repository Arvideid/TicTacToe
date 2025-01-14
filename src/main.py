import tkinter as tk
from tkinter import ttk, messagebox
from game import TicTacToeGame
from agents import HumanAgent, RandomAgent, MinimaxAgent
from rl_agent import QLearningAgent

class TicTacToeGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Tic Tac Toe")
        
        # Game instance
        self.game = TicTacToeGame()
        
        # Agents
        self.agents = {
            "Human": HumanAgent(),
            "Random": RandomAgent(),
            "Minimax": MinimaxAgent(),
            "Q-Learning": QLearningAgent()  # This should be loaded from a trained model
        }
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        # Player selection
        frame_top = ttk.Frame(self.window, padding="10")
        frame_top.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(frame_top, text="Player X:").grid(row=0, column=0, padx=5)
        self.player_x = ttk.Combobox(frame_top, values=list(self.agents.keys()))
        self.player_x.set("Human")
        self.player_x.grid(row=0, column=1, padx=5)
        
        ttk.Label(frame_top, text="Player O:").grid(row=0, column=2, padx=5)
        self.player_o = ttk.Combobox(frame_top, values=list(self.agents.keys()))
        self.player_o.set("Q-Learning")
        self.player_o.grid(row=0, column=3, padx=5)
        
        # Game board
        frame_board = ttk.Frame(self.window, padding="10")
        frame_board.grid(row=1, column=0)
        
        self.buttons = []
        for i in range(3):
            for j in range(3):
                button = ttk.Button(
                    frame_board,
                    text="",
                    width=10,
                    command=lambda row=i, col=j: self.make_move(row, col)
                )
                button.grid(row=i, column=j, padx=2, pady=2)
                self.buttons.append(button)
        
        # Reset button
        ttk.Button(
            self.window,
            text="New Game",
            command=self.reset_game
        ).grid(row=2, column=0, pady=10)
    
    def make_move(self, row, col):
        if not self.game.is_valid_move(row, col):
            return
        
        # Make the move
        self.game.make_move(row, col)
        self.update_board()
        
        if self.game.game_over:
            self.show_game_result()
            return
        
        # If next player is AI, make its move
        current_player = "X" if self.game.current_player == 1 else "O"
        player_selection = self.player_x if current_player == "X" else self.player_o
        
        if player_selection.get() != "Human":
            agent = self.agents[player_selection.get()]
            row, col = agent.get_move(self.game)
            self.game.make_move(row, col)
            self.update_board()
            
            if self.game.game_over:
                self.show_game_result()
    
    def update_board(self):
        for i in range(3):
            for j in range(3):
                value = self.game.board[i, j]
                text = "X" if value == 1 else "O" if value == -1 else ""
                self.buttons[i * 3 + j].configure(text=text)
    
    def show_game_result(self):
        if self.game.winner == 1:
            messagebox.showinfo("Game Over", "Player X wins!")
        elif self.game.winner == -1:
            messagebox.showinfo("Game Over", "Player O wins!")
        else:
            messagebox.showinfo("Game Over", "It's a tie!")
    
    def reset_game(self):
        self.game.reset()
        for button in self.buttons:
            button.configure(text="")
    
    def run(self):
        self.window.mainloop()

def main():
    gui = TicTacToeGUI()
    gui.run()

if __name__ == "__main__":
    main()