import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from game import TicTacToeGame
from agents import RandomAgent, MinimaxAgent as PerfectAgent, QLearningAgent

class TicTacToeGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Tic-Tac-Toe AI Arena")
        self.root.configure(bg='#f0f0f0')
        
        # Configure grid weights for responsiveness
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Game state
        self.game = TicTacToeGame()
        self.agents = {
            'human': None,
            'random': RandomAgent(),
            'perfect': PerfectAgent(),
            'learning': QLearningAgent(learning_rate=0.7, discount_factor=0.9, initial_epsilon=0.3, min_epsilon=0.1, epsilon_decay_rate=0.995)
        }
        self.games_played = 0
        self.win_rates = []
        
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for responsiveness
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_columnconfigure(2, weight=1)
        
        # Left panel for controls
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, padx=10, sticky=tk.N+tk.S+tk.W+tk.E)
        
        # Game settings
        settings_frame = ttk.LabelFrame(left_panel, text="Game Settings", padding="10")
        settings_frame.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N, pady=(0,10))
        
        # Player selection with better spacing
        ttk.Label(settings_frame, text="Player X:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.player_x = ttk.Combobox(settings_frame, values=['human', 'random', 'perfect', 'learning'], width=15)
        self.player_x.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        self.player_x.set('random')
        
        ttk.Label(settings_frame, text="Player O:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.player_o = ttk.Combobox(settings_frame, values=['human', 'random', 'perfect', 'learning'], width=15)
        self.player_o.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        self.player_o.set('learning')
        
        # Game controls
        ttk.Label(settings_frame, text="Games:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.num_games = ttk.Entry(settings_frame, width=10)
        self.num_games.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        self.num_games.insert(0, "100")
        
        ttk.Label(settings_frame, text="Delay (ms):").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.move_delay = ttk.Entry(settings_frame, width=10)
        self.move_delay.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        self.move_delay.insert(0, "100")
        
        # Q-Learning parameters
        qlearn_frame = ttk.LabelFrame(left_panel, text="Q-Learning Settings", padding="10")
        qlearn_frame.grid(row=1, column=0, sticky=tk.W+tk.E+tk.N, pady=(0,10))
        
        # Learning rate
        ttk.Label(qlearn_frame, text="Learning Rate:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.learning_rate = ttk.Entry(qlearn_frame, width=10)
        self.learning_rate.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        self.learning_rate.insert(0, "0.7")
        
        # Discount factor
        ttk.Label(qlearn_frame, text="Discount Factor:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.discount_factor = ttk.Entry(qlearn_frame, width=10)
        self.discount_factor.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        self.discount_factor.insert(0, "0.9")
        
        # Initial epsilon
        ttk.Label(qlearn_frame, text="Initial ε:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.initial_epsilon = ttk.Entry(qlearn_frame, width=10)
        self.initial_epsilon.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        self.initial_epsilon.insert(0, "0.3")
        
        # Min epsilon
        ttk.Label(qlearn_frame, text="Min ε:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.min_epsilon = ttk.Entry(qlearn_frame, width=10)
        self.min_epsilon.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        self.min_epsilon.insert(0, "0.1")
        
        # Epsilon decay
        ttk.Label(qlearn_frame, text="ε Decay Rate:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.epsilon_decay = ttk.Entry(qlearn_frame, width=10)
        self.epsilon_decay.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        self.epsilon_decay.insert(0, "0.995")
        
        # Control buttons
        button_frame = ttk.Frame(left_panel)
        button_frame.grid(row=2, column=0, sticky=tk.W+tk.E, pady=10)
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        
        self.start_btn = ttk.Button(button_frame, text="Start Match", command=self.start_match)
        self.start_btn.grid(row=0, column=0, padx=5, sticky=tk.W+tk.E)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop Match", command=self.stop_match)
        self.stop_btn.grid(row=0, column=1, padx=5, sticky=tk.W+tk.E)
        self.stop_btn['state'] = 'disabled'
        
        # Center panel for game board
        center_panel = ttk.Frame(main_frame)
        center_panel.grid(row=0, column=1, padx=20, sticky=tk.N+tk.S+tk.E+tk.W)
        for i in range(3):
            center_panel.grid_rowconfigure(i, weight=1)
            center_panel.grid_columnconfigure(i, weight=1)
        
        # Game board with improved styling
        self.board_buttons = []
        for i in range(3):
            for j in range(3):
                btn = ttk.Button(center_panel, text='', width=5, style='Game.TButton')
                btn.grid(row=i, column=j, padx=3, pady=3, ipadx=10, ipady=10, sticky=tk.W+tk.E+tk.N+tk.S)
                btn.configure(command=lambda row=i, col=j: self.make_move(row, col))
                self.board_buttons.append(btn)
        
        # Right panel for stats
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=2, padx=10, sticky=tk.N+tk.S+tk.E+tk.W)
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)
        
        # Performance graph
        self.fig = Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().grid(row=0, column=0, pady=(0,10), sticky=tk.W+tk.E+tk.N+tk.S)
        
        self.ax.set_title('Learning Agent Win Rate')
        self.ax.set_xlabel('Games Played')
        self.ax.set_ylabel('Win Rate')
        self.ax.set_ylim(0, 1)
        self.ax.grid(True)
        
        # Match history
        history_frame = ttk.LabelFrame(right_panel, text="Match History", padding="5")
        history_frame.grid(row=1, column=0, sticky=tk.W+tk.E+tk.N+tk.S)
        
        # Treeview with scrollbar
        self.history_tree = ttk.Treeview(history_frame, columns=('game', 'winner', 'result'), 
                                       show='headings', height=8)
        self.history_tree.heading('game', text='Game #')
        self.history_tree.heading('winner', text='Winner')
        self.history_tree.heading('result', text='Learning Agent')
        self.history_tree.column('game', width=60)
        self.history_tree.column('winner', width=60)
        self.history_tree.column('result', width=100)
        self.history_tree.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N+tk.S)
        
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        scrollbar.grid(row=0, column=1, sticky=tk.N+tk.S)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to start")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, padding=5)
        self.status_label.grid(row=1, column=0, columnspan=3, sticky=tk.W+tk.E, pady=10)
        
        self.is_running = False
        self.games_remaining = 0
        
    def update_board(self):
        for i in range(3):
            for j in range(3):
                value = self.game.board[i, j]
                text = 'X' if value == 1 else 'O' if value == -1 else ''
                self.board_buttons[i * 3 + j]['text'] = text
    
    def make_move(self, row, col):
        if not self.is_running:
            return
            
        current_player = 'X' if self.game.current_player == 1 else 'O'
        agent_type = self.player_x.get() if current_player == 'X' else self.player_o.get()
        
        if agent_type == 'human':
            if self.game.make_move(row, col):
                self.update_board()
                self.process_game_state()
    
    def make_ai_move(self):
        if not self.is_running:
            return
            
        current_player = 'X' if self.game.current_player == 1 else 'O'
        agent_type = self.player_x.get() if current_player == 'X' else self.player_o.get()
        
        if agent_type != 'human':
            agent = self.agents[agent_type]
            move = agent.get_move(self.game)
            if self.game.make_move(*move):
                self.update_board()
                self.process_game_state()
    
    def process_game_state(self):
        if self.game.game_over:
            self.games_played += 1
            self.games_remaining -= 1
            
            # Update win rate
            learning_is_x = self.player_x.get() == 'learning'
            learning_is_o = self.player_o.get() == 'learning'
            
            if learning_is_x or learning_is_o:
                learning_won = (learning_is_x and self.game.winner == 1) or (learning_is_o and self.game.winner == -1)
                if len(self.win_rates) > 0:
                    prev_wins = self.win_rates[-1] * (self.games_played - 1)
                    new_win_rate = (prev_wins + (1 if learning_won else 0)) / self.games_played
                else:
                    new_win_rate = 1 if learning_won else 0
                self.win_rates.append(new_win_rate)
                
                # Update graph
                self.ax.clear()
                self.ax.plot(range(1, len(self.win_rates) + 1), self.win_rates, 'b-')
                self.ax.set_title('Learning Agent Win Rate')
                self.ax.set_xlabel('Games Played')
                self.ax.set_ylabel('Win Rate')
                self.ax.set_ylim(0, 1)
                self.ax.grid(True)
                self.canvas.draw()
            
            # Update history
            winner = 'X' if self.game.winner == 1 else 'O' if self.game.winner == -1 else 'Tie'
            result = ''
            if learning_is_x or learning_is_o:
                if (learning_is_x and self.game.winner == 1) or (learning_is_o and self.game.winner == -1):
                    result = 'Win'
                elif self.game.winner == 0:
                    result = 'Tie'
                else:
                    result = 'Loss'
            
            self.history_tree.insert('', 0, values=(self.games_played, winner, result))
            
            # Keep only last 10 entries
            if len(self.history_tree.get_children()) > 10:
                self.history_tree.delete(self.history_tree.get_children()[-1])
            
            if self.games_remaining > 0:
                self.game = TicTacToeGame()
                self.update_board()
                self.status_var.set(f'Games remaining: {self.games_remaining}')
                self.root.after(int(self.move_delay.get()), self.make_ai_move)
            else:
                self.stop_match()
                self.status_var.set('Match complete!')
        else:
            next_player = 'X' if self.game.current_player == 1 else 'O'
            next_agent = self.player_x.get() if next_player == 'X' else self.player_o.get()
            if next_agent != 'human':
                self.root.after(int(self.move_delay.get()), self.make_ai_move)
    
    def start_match(self):
        try:
            # Validate Q-learning parameters
            lr = float(self.learning_rate.get())
            df = float(self.discount_factor.get())
            ie = float(self.initial_epsilon.get())
            me = float(self.min_epsilon.get())
            ed = float(self.epsilon_decay.get())
            
            if not (0 < lr <= 1 and 0 < df <= 1 and 0 <= ie <= 1 and 0 <= me <= ie and 0 < ed <= 1):
                raise ValueError("Invalid Q-learning parameters. All values must be between 0 and 1.")
            
            # Create new Q-learning agent with current parameters
            self.agents['learning'] = QLearningAgent(
                learning_rate=lr,
                discount_factor=df,
                initial_epsilon=ie,
                min_epsilon=me,
                epsilon_decay_rate=ed
            )
            
            self.games_remaining = int(self.num_games.get())
            if self.games_remaining <= 0:
                raise ValueError("Number of games must be positive")
                
            self.is_running = True
            self.start_btn['state'] = 'disabled'
            self.stop_btn['state'] = 'normal'
            self.player_x['state'] = 'disabled'
            self.player_o['state'] = 'disabled'
            
            self.game = TicTacToeGame()
            self.update_board()
            self.status_var.set(f'Games remaining: {self.games_remaining}')
            
            # Start AI moves if first player is AI
            first_player = self.player_x.get()
            if first_player != 'human':
                self.root.after(int(self.move_delay.get()), self.make_ai_move)
                
        except ValueError as e:
            self.status_var.set(str(e))
    
    def stop_match(self):
        self.is_running = False
        self.start_btn['state'] = 'normal'
        self.stop_btn['state'] = 'disabled'
        self.player_x['state'] = 'normal'
        self.player_o['state'] = 'normal'
        self.status_var.set('Match stopped')
    
    def run(self):
        # Configure style
        style = ttk.Style()
        style.configure('Game.TButton', font=('Arial', 14, 'bold'))
        
        # Center window
        self.root.update()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f'+{x}+{y}')
        
        self.root.mainloop()

if __name__ == '__main__':
    gui = TicTacToeGUI()
    gui.run()
