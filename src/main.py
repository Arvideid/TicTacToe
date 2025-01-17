"""
Main entry point for the Tic-Tac-Toe AI Arena application.
This module implements the graphical user interface and manages the game flow.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from game import TicTacToeGame
from agents import RandomAgent, MinimaxAgent, QLearningAgent, HumanAgent

class TicTacToeArena:
    """
    Main application class that provides a GUI for playing and analyzing Tic-Tac-Toe matches
    between different AI agents.
    """
    def __init__(self):
        # Initialize main window
        self.window = tk.Tk()
        self.window.title("Tic-Tac-Toe AI Arena")
        self.window.geometry("900x600")  # Adjusted size for a more compact layout
        
        # Initialize Q-learning parameters with default values
        self.learning_rate = tk.StringVar(value="0.4")
        self.discount_factor = tk.StringVar(value="0.95")
        self.initial_epsilon = tk.StringVar(value="0.9")
        self.min_epsilon = tk.StringVar(value="0.05")
        self.epsilon_decay_rate = tk.StringVar(value="0.997")
        
        # Initialize game instance and available AI agents
        self.game = TicTacToeGame()
        self.agents = {
            "Random AI": RandomAgent(),
            "Perfect AI": MinimaxAgent(),
            "Learning AI": QLearningAgent(
                learning_rate=self.learning_rate,
                discount_factor=self.discount_factor,
                initial_epsilon=self.initial_epsilon,
                min_epsilon=self.min_epsilon,
                epsilon_decay_rate=self.epsilon_decay_rate
            ),
            "Human": HumanAgent()
        }
        
        # Initialize statistics tracking
        self.session_stats = {
            'episodes': [],
            'vs_random': {'wins': [], 'ties': [], 'losses': []},
            'vs_perfect': {'wins': [], 'ties': [], 'losses': []}
        }
        
        # Match state variables
        self.is_match_running = False
        self.match_delay = 1000  # Default delay between moves (ms)
        self.current_match_stats = {"x_wins": 0, "o_wins": 0, "ties": 0}
        
        # Create the GUI layout
        self.create_layout()
        
        # Bind window resize event for responsive design
        self.window.bind("<Configure>", self.on_window_resize)
        
    def create_layout(self):
        """Create the main responsive layout with tabs."""
        # Main container with grid weights for responsiveness
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure weights for responsiveness
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Create tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        
        self.arena_tab = ttk.Frame(self.notebook)
        self.performance_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.arena_tab, text="Arena")
        self.notebook.add(self.performance_tab, text="Performance")
        
        # Setup panels in tabs
        self.setup_arena_panel(self.arena_tab)
        self.setup_stats_panel(self.performance_tab)
    
    def setup_arena_panel(self, parent):
        """Setup the arena panel with responsive layout."""
        # Create a horizontal frame to hold settings and game board
        horizontal_frame = ttk.Frame(parent)
        horizontal_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure horizontal frame to expand
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        horizontal_frame.grid_columnconfigure(0, weight=1)  # Settings frame
        horizontal_frame.grid_columnconfigure(1, weight=1)  # Game board frame
        
        # Create a frame for settings (left side)
        settings_frame = ttk.Frame(horizontal_frame)
        settings_frame.grid(row=0, column=0, sticky="nsew")
        
        # Create a canvas within the settings frame for scrollability
        canvas = tk.Canvas(settings_frame)
        canvas.grid(row=0, column=0, sticky="nsew")
        
        # Add a scrollbar to the canvas
        vscrollbar = ttk.Scrollbar(settings_frame, orient="vertical", command=canvas.yview)
        vscrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=vscrollbar.set)
        
        # Create a frame inside the canvas to hold the widgets
        arena_frame = ttk.LabelFrame(canvas, text="AI Arena", padding=10)
        canvas.create_window((0, 0), window=arena_frame, anchor="nw")
        
        # Configure the canvas to update scroll region
        arena_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Configure weights for responsiveness of settings frame
        settings_frame.grid_rowconfigure(0, weight=1)
        settings_frame.grid_columnconfigure(0, weight=1)
        
        # AI Selection
        select_frame = ttk.LabelFrame(arena_frame, text="Select AI Players", padding=10)
        select_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        select_frame.grid_columnconfigure(0, weight=1)
        
        # Player X selection
        player_x_frame = ttk.Frame(select_frame)
        player_x_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        player_x_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(player_x_frame, text="Player X (First):").grid(row=0, column=0, sticky="w", padx=5)
        self.player_x = ttk.Combobox(player_x_frame, values=list(self.agents.keys()), state="readonly")
        self.player_x.set("Random AI")
        self.player_x.grid(row=0, column=1, sticky="ew", padx=5)
        
        # Player O selection
        player_o_frame = ttk.Frame(select_frame)
        player_o_frame.grid(row=1, column=0, sticky="ew")
        player_o_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(player_o_frame, text="Player O (Second):").grid(row=0, column=0, sticky="w", padx=5)
        self.player_o = ttk.Combobox(player_o_frame, values=list(self.agents.keys()), state="readonly")
        self.player_o.set("Learning AI")
        self.player_o.grid(row=0, column=1, sticky="ew", padx=5)
        
        # Match controls
        controls_frame = ttk.LabelFrame(arena_frame, text="Match Controls", padding=10)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=10)
        controls_frame.grid_columnconfigure(0, weight=1)
        
        # Number of games with validation
        num_games_frame = ttk.Frame(controls_frame)
        num_games_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        num_games_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(num_games_frame, text="Number of Games:").grid(row=0, column=0, sticky="w", padx=5)
        vcmd = (self.window.register(self.validate_number), '%P')
        self.num_games = ttk.Spinbox(num_games_frame, from_=1, to=1000, width=10, validate='key', validatecommand=vcmd)
        self.num_games.set("100")
        self.num_games.grid(row=0, column=1, sticky="ew", padx=5)
        
        # Move delay with validation
        delay_frame = ttk.Frame(controls_frame)
        delay_frame.grid(row=1, column=0, sticky="ew")
        delay_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(delay_frame, text="Move Delay (ms):").grid(row=0, column=0, sticky="w", padx=5)
        self.delay_ms = ttk.Spinbox(delay_frame, from_=0, to=5000, width=10, validate='key', validatecommand=vcmd)
        self.delay_ms.set("10")
        self.delay_ms.grid(row=0, column=1, sticky="ew", padx=5)
        
        # Buttons frame
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.grid(row=2, column=0, pady=10)
        buttons_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Start/Stop buttons
        self.start_button = ttk.Button(buttons_frame, text="Start Match", command=self.start_match)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(buttons_frame, text="Stop Match", command=self.stop_match, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5)
        
        # Q-learning parameters
        ql_params_frame = ttk.LabelFrame(arena_frame, text="Q-Learning Parameters", padding=10)
        ql_params_frame.grid(row=4, column=0, sticky="ew", pady=10)
        ql_params_frame.grid_columnconfigure(0, weight=1)
        
        # Learning rate
        lr_frame = ttk.Frame(ql_params_frame)
        lr_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        lr_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(lr_frame, text="Learning Rate (α):").grid(row=0, column=0, sticky="w", padx=5)
        self.learning_rate = tk.StringVar(value="0.4")
        ttk.Entry(lr_frame, textvariable=self.learning_rate, width=10).grid(row=0, column=1, sticky="ew", padx=5)
        
        # Discount factor
        df_frame = ttk.Frame(ql_params_frame)
        df_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        df_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(df_frame, text="Discount Factor (γ):").grid(row=0, column=0, sticky="w", padx=5)
        self.discount_factor = tk.StringVar(value="0.95")
        ttk.Entry(df_frame, textvariable=self.discount_factor, width=10).grid(row=0, column=1, sticky="ew", padx=5)
        
        # Initial epsilon
        ie_frame = ttk.Frame(ql_params_frame)
        ie_frame.grid(row=2, column=0, sticky="ew", pady=(0, 5))
        ie_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(ie_frame, text="Initial Epsilon:").grid(row=0, column=0, sticky="w", padx=5)
        self.initial_epsilon = tk.StringVar(value="0.9")
        ttk.Entry(ie_frame, textvariable=self.initial_epsilon, width=10).grid(row=0, column=1, sticky="ew", padx=5)
        
        # Minimum epsilon
        me_frame = ttk.Frame(ql_params_frame)
        me_frame.grid(row=3, column=0, sticky="ew", pady=(0, 5))
        me_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(me_frame, text="Minimum Epsilon:").grid(row=0, column=0, sticky="w", padx=5)
        self.min_epsilon = tk.StringVar(value="0.05")
        ttk.Entry(me_frame, textvariable=self.min_epsilon, width=10).grid(row=0, column=1, sticky="ew", padx=5)
        
        # Epsilon decay rate
        edr_frame = ttk.Frame(ql_params_frame)
        edr_frame.grid(row=4, column=0, sticky="ew")
        edr_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Label(edr_frame, text="Epsilon Decay Rate:").grid(row=0, column=0, sticky="w", padx=5)
        self.epsilon_decay_rate = tk.StringVar(value="0.997")
        ttk.Entry(edr_frame, textvariable=self.epsilon_decay_rate, width=10).grid(row=0, column=1, sticky="ew", padx=5)
        
        # Create a frame for the game board (right side)
        board_frame = ttk.LabelFrame(horizontal_frame, text="Game Board", padding=10)
        board_frame.grid(row=0, column=1, sticky="nsew")
        
        # Configure weights to make the game board expand
        board_frame.grid_columnconfigure((0, 1, 2), weight=1)
        board_frame.grid_rowconfigure((0, 1, 2), weight=1)
        
        # Game board (responsive)
        self.buttons = []
        for i in range(3):
            for j in range(3):
                button = tk.Button(board_frame, text="", state="disabled", width=4, height=1, font=("Arial", 18))
                button.grid(row=i, column=j, sticky="nsew", padx=1, pady=1)
                self.buttons.append(button)
        
        # Status label
        self.status_label = ttk.Label(board_frame, text="", font=("Arial", 14))
        self.status_label.grid(row=3, column=0, columnspan=3, pady=10)
    
    def setup_stats_panel(self, parent):
        """Setup the statistics panel with responsive layout."""
        stats_frame = ttk.LabelFrame(parent, text="Statistics", padding=10)
        stats_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure weights for responsiveness
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        stats_frame.grid_columnconfigure(0, weight=1)
        
        # Current match stats
        match_stats_frame = ttk.LabelFrame(stats_frame, text="Current Match", padding=10)
        match_stats_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        match_stats_frame.grid_columnconfigure(1, weight=1)
        
        self.match_stats_labels = {}
        stats = [("X Wins:", "x_wins"), ("O Wins:", "o_wins"), ("Ties:", "ties")]
        for i, (text, key) in enumerate(stats):
            ttk.Label(match_stats_frame, text=text).grid(row=i, column=0, sticky="w", padx=5)
            self.match_stats_labels[key] = ttk.Label(match_stats_frame, text="0")
            self.match_stats_labels[key].grid(row=i, column=1, sticky="e", padx=5)
        
        # Performance plots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=stats_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, sticky="nsew")
        
        # History text
        history_frame = ttk.LabelFrame(stats_frame, text="History", padding=10)
        history_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        history_frame.grid_columnconfigure(0, weight=1)
        history_frame.grid_rowconfigure(0, weight=1)
        
        self.history_text = tk.Text(history_frame, wrap=tk.WORD, state="disabled", height=10)
        self.history_text.grid(row=0, column=0, sticky="nsew")
        
        # Add a scrollbar to the history text
        yscrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_text.yview)
        yscrollbar.grid(row=0, column=1, sticky="ns")
        self.history_text["yscrollcommand"] = yscrollbar.set
    
    def on_window_resize(self, event=None):
        """Handle window resize event for responsive design."""
        # Adjust padding and margins for a more compact layout
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Adjust font sizes dynamically based on window size
        base_font_size = 8
        new_font_size = max(base_font_size, int(base_font_size * self.window.winfo_width() / 800))
        default_font = ("Arial", new_font_size)
        button_font = ("Arial", new_font_size * 2)
        
        style = ttk.Style()
        style.configure('.', font=default_font)
        
        for button in self.buttons:
            button.config(font=button_font)
        self.status_label.config(font=default_font)
        self.history_text.config(font=default_font)

        # Adjust grid weights for responsiveness
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.arena_tab.grid_columnconfigure(0, weight=1)
        self.performance_tab.grid_columnconfigure(0, weight=1)
        
        # Adjust the game board to be more compact
        for i in range(3):
            for j in range(3):
                button = self.buttons[i * 3 + j]
                button.config(width=4, height=1, font=("Arial", 18))
                button.grid(row=i, column=j, sticky="nsew", padx=1, pady=1)
    
    def validate_number(self, value):
        """Validate that the input is a positive integer."""
        if value == "":
            return True
        try:
            int_value = int(value)
            if int_value >= 0:
                return True
            else:
                return False
        except ValueError:
            return False
    
    def update_match_stats(self):
        """Update the match statistics display."""
        for key, value in self.current_match_stats.items():
            self.match_stats_labels[key].config(text=str(value))
    
    def update_performance_plots(self):
        """Update the performance plots with the latest data."""
        self.ax1.clear()
        self.ax2.clear()
        
        episodes = self.session_stats['episodes']
        
        # Plot win rate against Random AI
        if len(self.session_stats['vs_random']['wins']) > 0:
            vs_random_wr = [w / (w + t + l) for w, t, l in zip(self.session_stats['vs_random']['wins'],
                                                               self.session_stats['vs_random']['ties'],
                                                               self.session_stats['vs_random']['losses'])]
            self.ax1.plot(episodes, vs_random_wr, label='vs Random', color='blue', marker='o')
        
        # Plot win rate against Perfect AI
        if len(self.session_stats['vs_perfect']['wins']) > 0:
            vs_perfect_wr = [w / (w + t + l) for w, t, l in zip(self.session_stats['vs_perfect']['wins'],
                                                                self.session_stats['vs_perfect']['ties'],
                                                                self.session_stats['vs_perfect']['losses'])]
            self.ax1.plot(episodes, vs_perfect_wr, label='vs Perfect', color='red', marker='x')
        
        # Plot non-loss rate against Random AI
        if len(self.session_stats['vs_random']['wins']) > 0:
            vs_random_nlr = [(w + t) / (w + t + l) for w, t, l in zip(self.session_stats['vs_random']['wins'],
                                                                      self.session_stats['vs_random']['ties'],
                                                                      self.session_stats['vs_random']['losses'])]
            self.ax2.plot(episodes, vs_random_nlr, label='vs Random', color='green', marker='o')
        
        # Plot non-loss rate against Perfect AI
        if len(self.session_stats['vs_perfect']['wins']) > 0:
            vs_perfect_nlr = [(w + t) / (w + t + l) for w, t, l in zip(self.session_stats['vs_perfect']['wins'],
                                                                       self.session_stats['vs_perfect']['ties'],
                                                                       self.session_stats['vs_perfect']['losses'])]
            self.ax2.plot(episodes, vs_perfect_nlr, label='vs Perfect', color='orange', marker='x')
        
        # Configure plots
        self.ax1.set_title('Win Rate vs Episodes')
        self.ax1.set_xlabel('Episodes')
        self.ax1.set_ylabel('Win Rate')
        self.ax1.legend()
        self.ax1.grid(True)
        
        self.ax2.set_title('Non-Loss Rate vs Episodes')
        self.ax2.set_xlabel('Episodes')
        self.ax2.set_ylabel('Non-Loss Rate')
        self.ax2.legend()
        self.ax2.grid(True)
        
        self.canvas.draw()
    
    def update_history_text(self):
        """Update the history text with the latest game results."""
        self.history_text.config(state="normal")
        self.history_text.delete("1.0", tk.END)
        
        for episode, (wins_r, ties_r, losses_r) in enumerate(zip(self.session_stats['vs_random']['wins'],
                                                                self.session_stats['vs_random']['ties'],
                                                                self.session_stats['vs_random']['losses'])):
            if (wins_r + ties_r + losses_r) > 0:
                vs_random_wr = wins_r / (wins_r + ties_r + losses_r)
                vs_random_nlr = (wins_r + ties_r) / (wins_r + ties_r + losses_r)
                
                self.history_text.insert(tk.END, f"Episode {episode} (vs Random):\n")
                self.history_text.insert(tk.END, f"  Wins: {wins_r}, Ties: {ties_r}, Losses: {losses_r}\n")
                self.history_text.insert(tk.END, f"  Win Rate: {vs_random_wr:.2%}, Non-Loss Rate: {vs_random_nlr:.2%}\n")
        
        for episode, (wins_p, ties_p, losses_p) in enumerate(zip(self.session_stats['vs_perfect']['wins'],
                                                                self.session_stats['vs_perfect']['ties'],
                                                                self.session_stats['vs_perfect']['losses'])):
            if (wins_p + ties_p + losses_p) > 0:
                vs_perfect_wr = wins_p / (wins_p + ties_p + losses_p)
                vs_perfect_nlr = (wins_p + ties_p) / (wins_p + ties_p + losses_p)
                
                self.history_text.insert(tk.END, f"Episode {episode} (vs Perfect):\n")
                self.history_text.insert(tk.END, f"  Wins: {wins_p}, Ties: {ties_p}, Losses: {losses_p}\n")
                self.history_text.insert(tk.END, f"  Win Rate: {vs_perfect_wr:.2%}, Non-Loss Rate: {vs_perfect_nlr:.2%}\n")
        
        self.history_text.config(state="disabled")
    
    def make_ai_move(self):
        """Make a move for the current AI player with real-time updates."""
        if not self.is_match_running:
            return
        
        if self.game.game_over:
            # Handle game over
            if self.game.winner == 1:
                self.current_match_stats["x_wins"] += 1
            elif self.game.winner == -1:
                self.current_match_stats["o_wins"] += 1
            else:
                self.current_match_stats["ties"] += 1
            
            # Update match stats immediately
            self.update_match_stats()
            
            # Update session stats if Learning AI is playing
            if "Learning AI" in [self.player_x.get(), self.player_o.get()]:
                is_learning_x = self.player_x.get() == "Learning AI"
                games_played = int(self.num_games.get()) - self.num_games_remaining + 1
                
                # Determine opponent and update stats
                opponent = self.player_o.get() if is_learning_x else self.player_x.get()
                stats_key = 'vs_random' if opponent == "Random AI" else 'vs_perfect'
                
                # Calculate result from Learning AI's perspective
                if is_learning_x:
                    won = self.game.winner == 1
                    lost = self.game.winner == -1
                else:
                    won = self.game.winner == -1
                    lost = self.game.winner == 1
                
                # Update session stats
                if games_played % 5 == 0:
                    self.session_stats['episodes'].append(games_played)
                    self.session_stats[stats_key]['wins'].append(self.session_stats[stats_key]['wins'][-1] + won if len(self.session_stats[stats_key]['wins']) > 0 else won)
                    self.session_stats[stats_key]['ties'].append(self.session_stats[stats_key]['ties'][-1] + (not won and not lost) if len(self.session_stats[stats_key]['ties']) > 0 else (not won and not lost))
                    self.session_stats[stats_key]['losses'].append(self.session_stats[stats_key]['losses'][-1] + lost if len(self.session_stats[stats_key]['losses']) > 0 else lost)
                    
                    # Update plots and history
                    self.update_performance_plots()
                    self.update_history_text()
            
            # Decrement games remaining and check if match is over
            self.num_games_remaining -= 1
            if self.num_games_remaining <= 0:
                self.stop_match()
                return
            
            # Start next game after delay
            self.window.after(self.match_delay, self.play_game)
            return
        
        # Get current player's agent
        current_player = "X" if self.game.current_player == 1 else "O"
        agent_type = self.player_x.get() if current_player == "X" else self.player_o.get()

        # If the current agent is human, enable buttons for input
        if agent_type == "Human":
            self.status_label.config(text=f"Your turn ({current_player})")
            for i in range(3):
                for j in range(3):
                    if self.game.board[i, j] == 0:
                        self.buttons[i * 3 + j].config(state="normal", command=lambda row=i, col=j: self.make_human_move(row, col))
            return  # Stop here and wait for human input via button click

        # For AI agents, proceed as before
        agent = self.agents[agent_type]
        row, col = agent.get_move(self.game)
        self.game.make_move(row, col)
        self.update_board()
        
        # Update status with current player and their type
        self.status_label.config(text=f"Current player: {current_player} ({agent_type})")
        
        # Schedule next move
        self.window.after(self.match_delay, self.make_ai_move)
    
    def update_board(self):
        """Update the game board display."""
        for i in range(3):
            for j in range(3):
                value = self.game.board[i, j]
                text = "X" if value == 1 else "O" if value == -1 else ""
                button = self.buttons[i * 3 + j]
                button.config(text=text, command=lambda: None)

                # Update button appearance based on game state
                if text:
                    # Button is no longer clickable, change its appearance
                    button.config(
                        state="disabled",
                        disabledforeground="black" if text == "X" else "blue",  # Change text color
                    )
                else:
                    # Reset appearance for empty cells
                    button.config(state="normal")
    
    def start_match(self):
        """Start an AI vs AI match."""
        global learning_role
        if self.is_match_running:
            return
        
        try:
            self.num_games_remaining = int(self.num_games.get())
            self.match_delay = int(self.delay_ms.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for games and delay.")
            return
        
        # Reset match stats
        self.current_match_stats = {"x_wins": 0, "o_wins": 0, "ties": 0}
        self.update_match_stats()
        
        # Update UI
        self.is_match_running = True
        self.start_button.state(['disabled'])
        self.stop_button.state(['!disabled'])
        self.player_x.state(['disabled'])
        self.player_o.state(['disabled'])
        
        # Determine which player is the Learning AI for plot labels
        learning_role = "X" if self.player_x.get() == "Learning AI" else "O"
        
        # Start the match
        self.play_game()
    
    def stop_match(self):
        """Stop the current match."""
        self.is_match_running = False
        self.start_button.state(['!disabled'])
        self.stop_button.state(['disabled'])
        self.player_x.state(['!disabled'])
        self.player_o.state(['!disabled'])
        self.status_label.config(text="Match stopped")
    
    def play_game(self):
        """Play a single game in the match."""
        if not self.is_match_running or self.num_games_remaining <= 0:
            self.stop_match()
            return
        
        # Reset game state
        self.game.reset()
        self.update_board()
        
        # Reset button appearance for new game
        for button in self.buttons:
            button.config(state="normal")
        
        # Make AI move
        self.make_ai_move()
    
    def run(self):
        """Start the application."""
        self.window.mainloop()

    def make_human_move(self, row, col):
        """Process human move and update the game."""
        # Make the move on the board
        self.game.make_move(row, col)
        self.update_board()

        # Continue with the game immediately after human move
        self.make_ai_move()

def main():
    arena = TicTacToeArena()
    arena.run()

if __name__ == "__main__":
    main()