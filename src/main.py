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
        self.window.geometry("1200x800")
        
        # Initialize game instance and available AI agents
        self.game = TicTacToeGame()
        self.agents = {
            "Random AI": RandomAgent(),      # Makes random valid moves
            "Perfect AI": MinimaxAgent(),    # Uses minimax algorithm for perfect play
            "Learning AI": QLearningAgent(),  # Uses Q-learning to improve over time
            "Human": HumanAgent()  # Add HumanAgent to the available agents
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
        """Create the main responsive layout."""
        # Main container with grid weights for responsiveness
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Configure weights for responsiveness
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=3)  # Stats panel takes more space
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Create panels
        self.setup_arena_panel()
        self.setup_stats_panel()
    
    def setup_arena_panel(self):
        """Setup the left panel with responsive layout."""
        arena_frame = ttk.LabelFrame(self.main_frame, text="AI Arena", padding=10)
        arena_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        arena_frame.grid_columnconfigure(0, weight=1)
        
        # AI Selection
        select_frame = ttk.LabelFrame(arena_frame, text="Select AI Players", padding=10)
        select_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        select_frame.grid_columnconfigure(1, weight=1)
        
        # Player X selection
        ttk.Label(select_frame, text="Player X (First):").grid(row=0, column=0, sticky="w", padx=5)
        self.player_x = ttk.Combobox(select_frame, values=list(self.agents.keys()), state="readonly")
        self.player_x.set("Random AI")
        self.player_x.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # Player O selection
        ttk.Label(select_frame, text="Player O (Second):").grid(row=1, column=0, sticky="w", padx=5)
        self.player_o = ttk.Combobox(select_frame, values=list(self.agents.keys()), state="readonly")
        self.player_o.set("Learning AI")
        self.player_o.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Match controls
        controls_frame = ttk.LabelFrame(arena_frame, text="Match Controls", padding=10)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=10)
        controls_frame.grid_columnconfigure(1, weight=1)
        
        # Number of games with validation
        ttk.Label(controls_frame, text="Number of Games:").grid(row=0, column=0, sticky="w", padx=5)
        vcmd = (self.window.register(self.validate_number), '%P')
        self.num_games = ttk.Spinbox(controls_frame, from_=1, to=1000, width=10, validate='key', validatecommand=vcmd)
        self.num_games.set("100")
        self.num_games.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # Move delay with validation
        ttk.Label(controls_frame, text="Move Delay (ms):").grid(row=1, column=0, sticky="w", padx=5)
        self.delay_ms = ttk.Spinbox(controls_frame, from_=0, to=5000, width=10, validate='key', validatecommand=vcmd)
        self.delay_ms.set("10")
        self.delay_ms.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Buttons frame
        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.grid(row=2, column=0, columnspan=2, pady=10)
        buttons_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Start/Stop buttons
        self.start_button = ttk.Button(buttons_frame, text="Start Match", command=self.start_match)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(buttons_frame, text="Stop Match", command=self.stop_match, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5)
        
        # Game board (responsive)
        board_frame = ttk.LabelFrame(arena_frame, text="Game Board", padding=10)
        board_frame.grid(row=2, column=0, sticky="nsew", pady=10)
        board_frame.grid_columnconfigure((0, 1, 2), weight=1)
        board_frame.grid_rowconfigure((0, 1, 2), weight=1)
        
        # Make board buttons square and responsive
        self.buttons = []
        for i in range(3):
            for j in range(3):
                button = tk.Button(board_frame, text="", state="disabled", width=5, height=2, font=("Arial", 24))
                button.grid(row=i, column=j, sticky="nsew", padx=2, pady=2)
                self.buttons.append(button)
        
        # Match status with dynamic font
        self.status_label = ttk.Label(
            arena_frame,
            text="Ready to start match",
            font=("Arial", 12, "bold"),
            wraplength=200  # Wrap text if window is narrow
        )
        self.status_label.grid(row=3, column=0, pady=10)
        
        # Current match stats (responsive)
        stats_frame = ttk.LabelFrame(arena_frame, text="Current Match Stats", padding=10)
        stats_frame.grid(row=4, column=0, sticky="ew", pady=10)
        stats_frame.grid_columnconfigure(1, weight=1)
        
        self.match_stats_labels = {}
        stats = [
            ("Games Played:", "games"),
            ("X Wins:", "x_wins"),
            ("O Wins:", "o_wins"),
            ("Ties:", "ties"),
            ("Win Rate:", "win_rate")
        ]
        
        for i, (text, key) in enumerate(stats):
            ttk.Label(stats_frame, text=text).grid(row=i, column=0, sticky="w", padx=5)
            self.match_stats_labels[key] = ttk.Label(stats_frame, text="0")
            self.match_stats_labels[key].grid(row=i, column=1, sticky="e", padx=5)
    
    def setup_stats_panel(self):
        """Setup the right panel with responsive graphs using tabs."""
        stats_frame = ttk.LabelFrame(self.main_frame, text="Performance Analysis", padding=10)
        stats_frame.grid(row=0, column=1, sticky="nsew")
        stats_frame.grid_columnconfigure(0, weight=1)
        stats_frame.grid_rowconfigure(0, weight=1)
        
        # Create notebook for tabs
        self.stats_notebook = ttk.Notebook(stats_frame)
        self.stats_notebook.grid(row=0, column=0, sticky="nsew")
        
        # Create tabs
        self.learning_ai_tab = ttk.Frame(self.stats_notebook)
        self.x_player_tab = ttk.Frame(self.stats_notebook)
        self.o_player_tab = ttk.Frame(self.stats_notebook)
        
        self.stats_notebook.add(self.learning_ai_tab, text="Learning AI")
        self.stats_notebook.add(self.x_player_tab, text="Player X")
        self.stats_notebook.add(self.o_player_tab, text="Player O")
        
        # Setup Learning AI tab
        self.setup_learning_ai_tab()
        
        # Setup X Player tab
        self.setup_player_tab(self.x_player_tab, "X")
        
        # Setup O Player tab
        self.setup_player_tab(self.o_player_tab, "O")
    
    def setup_learning_ai_tab(self):
        """Setup the Learning AI performance tab."""
        # Configure grid
        self.learning_ai_tab.grid_columnconfigure(0, weight=1)
        self.learning_ai_tab.grid_rowconfigure(0, weight=3)
        self.learning_ai_tab.grid_rowconfigure(1, weight=1)
        
        # Graphs container
        graphs_frame = ttk.Frame(self.learning_ai_tab)
        graphs_frame.grid(row=0, column=0, sticky="nsew", pady=(10, 10))
        graphs_frame.grid_columnconfigure(0, weight=1)
        graphs_frame.grid_rowconfigure(0, weight=1)
        
        # Create matplotlib figure for performance tracking
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graphs_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Initialize plots
        self.update_performance_plots()
        
        # Historical stats
        history_frame = ttk.LabelFrame(self.learning_ai_tab, text="Historical Performance", padding=10)
        history_frame.grid(row=1, column=0, sticky="nsew")
        history_frame.grid_columnconfigure(0, weight=1)
        history_frame.grid_rowconfigure(0, weight=1)
        
        # Scrollable text widget
        self.history_text = tk.Text(history_frame, wrap=tk.WORD, height=8)
        scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=scrollbar.set)
        
        self.history_text.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.update_history_text()
    
    def setup_player_tab(self, tab, player):
        """Setup a player-specific performance tab."""
        # Configure grid
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        
        # Stats container
        stats_container = ttk.Frame(tab)
        stats_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        stats_container.grid_columnconfigure(0, weight=1)
        
        # Create figure for player stats
        fig = plt.Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=stats_container)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Store references for updates
        if player == "X":
            self.x_fig = fig
            self.x_ax = ax
            self.x_canvas = canvas
        else:
            self.o_fig = fig
            self.o_ax = ax
            self.o_canvas = canvas
        
        # Initialize player plot
        self.update_player_plot(player)
    
    def update_player_plot(self, player):
        """Update the performance plot for a specific player."""
        ax = self.x_ax if player == "X" else self.o_ax
        fig = self.x_fig if player == "X" else self.o_fig
        canvas = self.x_canvas if player == "X" else self.o_canvas
        
        # Clear previous plot
        ax.clear()
        
        # Get total games
        total_games = sum(self.current_match_stats.values())
        
        if total_games > 0:
            # Calculate win rate over time
            wins = self.current_match_stats[f"{player.lower()}_wins"]
            win_rate = (wins / total_games) * 100
            
            # Create bar chart
            labels = ['Wins', 'Losses', 'Ties']
            if player == "X":
                values = [
                    self.current_match_stats["x_wins"],
                    self.current_match_stats["o_wins"],
                    self.current_match_stats["ties"]
                ]
            else:
                values = [
                    self.current_match_stats["o_wins"],
                    self.current_match_stats["x_wins"],
                    self.current_match_stats["ties"]
                ]
            
            colors = ['#2196F3', '#F44336', '#4CAF50']
            ax.bar(labels, values, color=colors)
            
            # Add win rate text
            ax.text(0.5, 0.95, f'Win Rate: {win_rate:.1f}%',
                   horizontalalignment='center',
                   transform=ax.transAxes,
                   fontsize=12,
                   fontweight='bold')
            
            # Configure plot
            ax.set_title(f'Player {player} Performance', pad=20, fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Games', fontsize=10)
            
            # Add value labels on top of bars
            for i, v in enumerate(values):
                ax.text(i, v, str(v), ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, "No games played yet", ha='center', va='center',
                   fontsize=12, transform=ax.transAxes)
        
        # Update canvas
        canvas.draw()
    
    def validate_number(self, value):
        """Validate numeric input for spinboxes."""
        if value == "":
            return True
        try:
            int(value)
            return True
        except ValueError:
            return False
    
    def on_window_resize(self, event):
        """Handle window resize events."""
        # Only handle main window resizes
        if event.widget == self.window:
            # Update plot size
            self.fig.set_size_inches(
                event.width * 0.4 / self.fig.get_dpi(),
                event.height * 0.5 / self.fig.get_dpi()
            )
            self.canvas.draw()
            
            # Update status label wraplength
            self.status_label.configure(wraplength=event.width * 0.2)
    
    def update_match_stats(self):
        """Update all statistics displays."""
        total_games = sum(self.current_match_stats.values())
        if total_games > 0:
            win_rate = (self.current_match_stats["x_wins"] / total_games) * 100
            self.match_stats_labels["win_rate"].config(text=f"{win_rate:.1f}%")
        
        self.match_stats_labels["games"].config(text=str(total_games))
        self.match_stats_labels["x_wins"].config(text=str(self.current_match_stats["x_wins"]))
        self.match_stats_labels["o_wins"].config(text=str(self.current_match_stats["o_wins"]))
        self.match_stats_labels["ties"].config(text=str(self.current_match_stats["ties"]))
        
        # Update all plots
        if total_games % 10 == 0:  # Update every 10 games for performance
            self.update_performance_plots()
            self.update_history_text()
            self.update_player_plot("X")
            self.update_player_plot("O")
    
    def update_performance_plots(self):
        """Update the performance analysis plots with real-time data."""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Only plot if we have data
        if len(self.session_stats['episodes']) > 0:
            # Get Learning AI's role (X or O)
            is_learning_x = self.player_x.get() == "Learning AI"
            learning_role = "X" if is_learning_x else "O"
            opponent_role = "O" if is_learning_x else "X"
            
            # Ensure all arrays have the same length
            num_episodes = len(self.session_stats['episodes'])
            episodes = self.session_stats['episodes'][-num_episodes:]
            
            # Calculate win rates for random opponent
            if len(self.session_stats['vs_random']['wins']) > 0:
                # Ensure arrays are of the same length
                wins = self.session_stats['vs_random']['wins'][-num_episodes:]
                ties = self.session_stats['vs_random']['ties'][-num_episodes:]
                losses = self.session_stats['vs_random']['losses'][-num_episodes:]
                
                vs_random_wr = [w/(w+t+l) if (w+t+l) > 0 else 0 
                               for w, t, l in zip(wins, ties, losses)]
                
                # Win rates over time
                self.ax1.plot(episodes, vs_random_wr, 
                             label=f'vs Random ({opponent_role})', color='#2196F3', linewidth=2)
                
                # Non-loss rate (wins + ties)
                vs_random_nlr = [(w+t)/(w+t+l) if (w+t+l) > 0 else 0 
                                for w, t, l in zip(wins, ties, losses)]
                
                self.ax2.plot(episodes, vs_random_nlr,
                             label=f'vs Random ({opponent_role})', color='#4CAF50', linewidth=2)
            
            # Add perfect opponent data if it exists
            if len(self.session_stats['vs_perfect']['wins']) > 0:
                # Ensure arrays are of the same length
                wins = self.session_stats['vs_perfect']['wins'][-num_episodes:]
                ties = self.session_stats['vs_perfect']['ties'][-num_episodes:]
                losses = self.session_stats['vs_perfect']['losses'][-num_episodes:]
                
                vs_perfect_wr = [w/(w+t+l) if (w+t+l) > 0 else 0 
                                for w, t, l in zip(wins, ties, losses)]
                
                self.ax1.plot(episodes, vs_perfect_wr,
                             label=f'vs Perfect ({opponent_role})', color='#F44336', linewidth=2)
                
                # Non-loss rate for perfect opponent
                vs_perfect_nlr = [(w+t)/(w+t+l) if (w+t+l) > 0 else 0 
                                 for w, t, l in zip(wins, ties, losses)]
                
                self.ax2.plot(episodes, vs_perfect_nlr,
                             label=f'vs Perfect ({opponent_role})', color='#FF9800', linewidth=2)
            
            # Configure plots
            self.ax1.set_title(f'Learning AI Performance (Playing as {learning_role})\nWin Rate Over Time', 
                             pad=10, fontsize=10, fontweight='bold')
            self.ax1.set_ylabel('Win Rate', fontsize=9)
            self.ax1.set_xlabel('Training Episodes', fontsize=9)
            self.ax1.grid(True, linestyle='--', alpha=0.7)
            self.ax1.legend(fontsize=8, loc='upper left')
            self.ax1.set_ylim(0, 1)
            
            self.ax2.set_title(f'Learning AI Success Rate (Win + Tie)\nPlaying as {learning_role}', 
                             pad=10, fontsize=10, fontweight='bold')
            self.ax2.set_xlabel('Training Episodes', fontsize=9)
            self.ax2.set_ylabel('Non-loss Rate', fontsize=9)
            self.ax2.grid(True, linestyle='--', alpha=0.7)
            self.ax2.legend(fontsize=8, loc='upper left')
            self.ax2.set_ylim(0, 1)
        else:
            self.ax1.text(0.5, 0.5, "Waiting for training data...", 
                         ha='center', va='center', fontsize=10)
            self.ax2.text(0.5, 0.5, "Waiting for training data...", 
                         ha='center', va='center', fontsize=10)
        
        plt.tight_layout()
        self.canvas.draw()
    
    def update_history_text(self):
        """Update the historical performance text with current session data."""
        self.history_text.delete(1.0, tk.END)
        
        if len(self.session_stats['episodes']) == 0:
            self.history_text.insert(tk.END, "No training data available in current session.")
            return
        
        # Get Learning AI's role
        is_learning_x = self.player_x.get() == "Learning AI"
        learning_role = "X" if is_learning_x else "O"
        opponent_role = "O" if is_learning_x else "X"
        
        self.history_text.insert(tk.END, f"Learning AI Training Summary\n", "heading")
        self.history_text.insert(tk.END, "=" * 40 + "\n\n")
        self.history_text.insert(tk.END, f"Playing as: {learning_role}\n", "subheading")
        self.history_text.insert(tk.END, "-" * 20 + "\n\n")
        
        # Calculate latest stats for both opponents
        for opponent, title in [('vs_random', 'Random AI'), ('vs_perfect', 'Perfect AI')]:
            if not self.session_stats[opponent]['wins']:  # Skip if no data
                continue
                
            total = sum(x[-1] if len(x) > 0 else 0 for x in [
                self.session_stats[opponent]['wins'],
                self.session_stats[opponent]['ties'],
                self.session_stats[opponent]['losses']
            ])
            
            if total > 0:
                wins = self.session_stats[opponent]['wins'][-1]
                ties = self.session_stats[opponent]['ties'][-1]
                losses = self.session_stats[opponent]['losses'][-1]
                
                self.history_text.insert(tk.END, f"Against {title} ({opponent_role}):\n", "subheading")
                self.history_text.insert(tk.END, f"• Win Rate:     {wins/total:.1%}\n")
                self.history_text.insert(tk.END, f"• Tie Rate:     {ties/total:.1%}\n")
                self.history_text.insert(tk.END, f"• Loss Rate:    {losses/total:.1%}\n")
                self.history_text.insert(tk.END, f"• Total Games:  {total}\n\n")
        
        # Add training progress
        episodes = self.session_stats['episodes'][-1]
        self.history_text.insert(tk.END, f"Training Progress:\n", "subheading")
        self.history_text.insert(tk.END, f"• Episodes Completed: {episodes:,}\n")
        
        # Configure tags for styling
        self.history_text.tag_configure("heading", font=("Arial", 11, "bold"))
        self.history_text.tag_configure("subheading", font=("Arial", 10, "bold"))
        self.history_text.configure(state="disabled")
    
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
                tied = self.game.winner == 0
                
                # Update cumulative stats every 5 games
                if games_played % 5 == 0:
                    # Add new game count
                    self.session_stats['episodes'].append(games_played)
                    
                    prev_wins = self.session_stats[stats_key]['wins'][-1] if self.session_stats[stats_key]['wins'] else 0
                    prev_ties = self.session_stats[stats_key]['ties'][-1] if self.session_stats[stats_key]['ties'] else 0
                    prev_losses = self.session_stats[stats_key]['losses'][-1] if self.session_stats[stats_key]['losses'] else 0
                    
                    self.session_stats[stats_key]['wins'].append(prev_wins + (1 if won else 0))
                    self.session_stats[stats_key]['ties'].append(prev_ties + (1 if tied else 0))
                    self.session_stats[stats_key]['losses'].append(prev_losses + (1 if lost else 0))
            
            self.num_games_remaining -= 1
            
            # Update status with more detailed information
            games_played = int(self.num_games.get()) - self.num_games_remaining
            total_games = int(self.num_games.get())
            progress = (games_played / total_games) * 100
            
            # Show who won
            winner_text = "X wins!" if self.game.winner == 1 else "O wins!" if self.game.winner == -1 else "Tie!"
            self.status_label.config(
                text=f"Game {games_played}/{total_games} - {winner_text} ({progress:.1f}%)"
            )
            
            # Update plots and history more frequently
            if games_played % 5 == 0:  # Update every 5 games
                self.update_performance_plots()
                self.update_history_text()
            
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