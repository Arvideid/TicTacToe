import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from game import TicTacToeGame
from agents import RandomAgent, MinimaxAgent, QLearningAgent
import os
import pickle
import datetime

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
            'perfect': MinimaxAgent(),
            'qlearning': QLearningAgent(
                learning_rate=0.7,
                discount_factor=0.9,
                initial_epsilon=0.3,
                min_epsilon=0.1,
                epsilon_decay_rate=0.995
            )
        }
        self.games_played = 0
        self.x_wins = []  # Track X player wins
        self.o_wins = []  # Track O player wins
        self.games_remaining = 0
        
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
        self.player_x = ttk.Combobox(settings_frame, values=['human', 'random', 'perfect', 'qlearning'], width=15)
        self.player_x.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        self.player_x.set('random')
        
        ttk.Label(settings_frame, text="Player O:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.player_o = ttk.Combobox(settings_frame, values=['human', 'random', 'perfect', 'qlearning'], width=15)
        self.player_o.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        self.player_o.set('qlearning')
        
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
        qlearn_frame = ttk.LabelFrame(left_panel, text="Learning Agent Settings", padding="10")
        qlearn_frame.grid(row=1, column=0, sticky=tk.W+tk.E+tk.N, pady=(0,10))
        
        # Learning type
        ttk.Label(qlearn_frame, text="Learning Type:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.learning_type = ttk.Combobox(qlearn_frame, values=['Q-Learning'], width=15)
        self.learning_type.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        self.learning_type.set('Q-Learning')
        
        # Basic parameters (common to both)
        ttk.Label(qlearn_frame, text="Learning Rate:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.learning_rate = ttk.Entry(qlearn_frame, width=10)
        self.learning_rate.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        self.learning_rate.insert(0, "0.7")
        
        ttk.Label(qlearn_frame, text="Discount Factor:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.discount_factor = ttk.Entry(qlearn_frame, width=10)
        self.discount_factor.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        self.discount_factor.insert(0, "0.9")
        
        # Epsilon parameters
        ttk.Label(qlearn_frame, text="Initial ε:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.initial_epsilon = ttk.Entry(qlearn_frame, width=10)
        self.initial_epsilon.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        self.initial_epsilon.insert(0, "0.3")
        
        ttk.Label(qlearn_frame, text="Min ε:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.min_epsilon = ttk.Entry(qlearn_frame, width=10)
        self.min_epsilon.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        self.min_epsilon.insert(0, "0.1")
        
        ttk.Label(qlearn_frame, text="ε Decay Rate:").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        self.epsilon_decay = ttk.Entry(qlearn_frame, width=10)
        self.epsilon_decay.grid(row=5, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
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
        
        self.save_btn = ttk.Button(button_frame, text="Save Agent", command=self.save_agent)
        self.save_btn.grid(row=1, column=0, padx=5, pady=(5,0), sticky=tk.W+tk.E)
        
        self.load_btn = ttk.Button(button_frame, text="Load Agent", command=self.load_agent)
        self.load_btn.grid(row=1, column=1, padx=5, pady=(5,0), sticky=tk.W+tk.E)
        
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
        
        self.ax.set_title('Win Rates Over Time')
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
        # Get learning agents if present
        x_is_learning = self.player_x.get() == 'qlearning'
        o_is_learning = self.player_o.get() == 'qlearning'
        
        x_agent = self.agents['qlearning'] if x_is_learning else None
        o_agent = self.agents['qlearning'] if o_is_learning else None
        
        if self.game.game_over:
            # Learning step for game end
            if x_agent:
                current_state = x_agent.board_to_state(self.game.board)
                reward = 5.0 if self.game.winner == 1 else (1.0 if self.game.winner == 0 else -3.0)
                x_agent.learn(current_state, x_agent.current_move, reward, None, [])
            
            if o_agent:
                current_state = o_agent.board_to_state(self.game.board)
                reward = 5.0 if self.game.winner == -1 else (1.0 if self.game.winner == 0 else -3.0)
                o_agent.learn(current_state, o_agent.current_move, reward, None, [])
            
            self.games_played += 1
            
            # Update win rates for both players
            if len(self.x_wins) > 0:
                prev_x_wins = self.x_wins[-1] * (self.games_played - 1)
                prev_o_wins = self.o_wins[-1] * (self.games_played - 1)
                
                # Calculate new win counts
                new_x_wins = prev_x_wins + (1 if self.game.winner == 1 else 0)
                new_o_wins = prev_o_wins + (1 if self.game.winner == -1 else 0)
                
                # Calculate new win rates
                self.x_wins.append(new_x_wins / self.games_played)
                self.o_wins.append(new_o_wins / self.games_played)
            else:
                # First game
                self.x_wins.append(1 if self.game.winner == 1 else 0)
                self.o_wins.append(1 if self.game.winner == -1 else 0)
            
            self.update_win_rate_graph()
            
            # Update history
            winner = 'X' if self.game.winner == 1 else 'O' if self.game.winner == -1 else 'Tie'
            result = winner
            
            self.history_tree.insert('', 0, values=(self.games_played, winner, result))
            
            # Keep only last 10 entries
            if len(self.history_tree.get_children()) > 10:
                self.history_tree.delete(self.history_tree.get_children()[-1])
            
            if self.games_remaining > 0:
                self.games_remaining -= 1
                self.game = TicTacToeGame()
                self.update_board()
                self.status_var.set(f'Games remaining: {self.games_remaining}')
                self.root.after(int(self.move_delay.get()), self.make_ai_move)
            else:
                self.stop_match()
                self.status_var.set('Match complete!')
        else:
            # Learning step for non-terminal states
            current_player = self.game.current_player
            current_agent = x_agent if current_player == 1 else o_agent
            
            if current_agent:
                current_state = current_agent.board_to_state(self.game.board)
                reward = current_agent.calculate_reward(self.game, current_state)
                next_valid_moves = self.game.get_valid_moves()
                current_agent.learn(current_state, current_agent.current_move, reward, current_state, next_valid_moves)
            
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
                raise ValueError("Invalid learning parameters. All values must be between 0 and 1.")
            
            # Create new learning agents with current parameters
            self.agents['qlearning'] = QLearningAgent(
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
    
    def save_agent(self):
        try:
            agent_type = self.learning_type.get()
            
            # Create save directory if it doesn't exist
            save_dir = 'saved_agents'
            os.makedirs(save_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Configure file dialog
            filetypes = [('Pickle files', '*.pkl'), ('All files', '*.*')]
            default_ext = '.pkl'
            initial_file = f'qlearning_agent_{timestamp}.pkl'
            
            # Open save file dialog
            save_path = filedialog.asksaveasfilename(
                initialdir=save_dir,
                initialfile=initial_file,
                defaultextension=default_ext,
                filetypes=filetypes,
                title='Save Q-Learning Agent'
            )
            
            if not save_path:  # User cancelled
                return
            
            # Ensure save directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save Q-table, parameters, and training history
            agent = self.agents['qlearning']
            save_data = {
                'q_table': agent.q_table,
                'params': {
                    'learning_rate': agent.learning_rate,
                    'discount_factor': agent.gamma,
                    'epsilon': agent.epsilon,
                    'min_epsilon': agent.min_epsilon,
                    'epsilon_decay_rate': agent.epsilon_decay
                },
                'training_history': {
                    'x_wins': self.x_wins,
                    'o_wins': self.o_wins,
                    'games_played': self.games_played,
                    'timestamp': timestamp
                }
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            messagebox.showinfo("Success", 
                f"{agent_type} agent saved successfully!\nLocation: {save_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save agent: {str(e)}")
    
    def load_agent(self):
        try:
            agent_type = self.learning_type.get()
            save_dir = 'saved_agents'
            
            # Configure file dialog
            filetypes = [('Pickle files', '*.pkl'), ('All files', '*.*')]
            title = 'Load Q-Learning Agent'
            
            # Open load file dialog
            load_path = filedialog.askopenfilename(
                initialdir=save_dir,
                filetypes=filetypes,
                title=title
            )
            
            if not load_path:  # User cancelled
                return
            
            with open(load_path, 'rb') as f:
                save_data = pickle.load(f)
            
            agent = QLearningAgent(
                learning_rate=save_data['params']['learning_rate'],
                discount_factor=save_data['params']['discount_factor'],
                initial_epsilon=save_data['params']['epsilon'],
                min_epsilon=save_data['params']['min_epsilon'],
                epsilon_decay_rate=save_data['params']['epsilon_decay_rate']
            )
            agent.q_table = save_data['q_table']
            self.agents['qlearning'] = agent
            
            # Restore training history if available
            if 'training_history' in save_data:
                if 'x_wins' in save_data['training_history']:
                    self.x_wins = save_data['training_history']['x_wins']
                    self.o_wins = save_data['training_history']['o_wins']
                else:
                    # For backward compatibility with old save files
                    self.x_wins = save_data['training_history']['win_rates']
                    self.o_wins = [0] * len(self.x_wins)
                
                self.games_played = save_data['training_history']['games_played']
                self.update_win_rate_graph()
            
            # Show detailed agent information
            self.show_agent_info(
                agent=self.agents['qlearning'],
                agent_type=agent_type,
                file_path=load_path
            )
            
            messagebox.showinfo("Success", 
                f"{agent_type} agent loaded successfully!\nLocation: {load_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load agent: {str(e)}")
    
    def show_agent_info(self, agent, agent_type, file_path):
        """Display a window with detailed information about the loaded agent."""
        info_window = tk.Toplevel(self.root)
        info_window.title(f"{agent_type} Agent Info")
        info_window.geometry("600x400")
        
        # Create a frame with scrollbar
        main_frame = ttk.Frame(info_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add a canvas
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add info labels
        ttk.Label(scrollable_frame, text="Agent Information", font=('Helvetica', 12, 'bold')).pack(pady=(0,10))
        
        # File info
        ttk.Label(scrollable_frame, text="File Location:", font=('Helvetica', 10, 'bold')).pack(anchor="w")
        ttk.Label(scrollable_frame, text=file_path, wraplength=550).pack(anchor="w", padx=10)
        
        # Parameters section
        ttk.Label(scrollable_frame, text="\nParameters:", font=('Helvetica', 10, 'bold')).pack(anchor="w", pady=(10,5))
        param_frame = ttk.Frame(scrollable_frame)
        param_frame.pack(fill=tk.X, padx=10)
        
        # Common parameters
        params = {
            'Learning Rate': agent.learning_rate,
            'Discount Factor': agent.gamma,
            'Current Epsilon': agent.epsilon,
            'Min Epsilon': agent.min_epsilon,
            'Epsilon Decay': agent.epsilon_decay
        }
        
        # Create parameter grid
        for i, (param, value) in enumerate(params.items()):
            ttk.Label(param_frame, text=f"{param}:", font=('Helvetica', 9, 'bold')).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Label(param_frame, text=str(value)).grid(row=i, column=1, sticky="w", padx=10, pady=2)
        
        # State information section
        ttk.Label(scrollable_frame, text="\nState Information:", font=('Helvetica', 10, 'bold')).pack(anchor="w", pady=(10,5))
        
        # Q-table statistics
        q_table_info = ttk.Frame(scrollable_frame)
        q_table_info.pack(fill=tk.X, padx=10)
        
        states_count = len(agent.q_table)
        total_actions = sum(len(actions) for actions in agent.q_table.values())
        avg_value = sum(sum(actions.values()) for actions in agent.q_table.values()) / total_actions if total_actions > 0 else 0
        
        stats = {
            'Total States Learned': states_count,
            'Total State-Actions': total_actions,
            'Average Q-Value': f"{avg_value:.4f}",
            'Non-zero States': sum(1 for actions in agent.q_table.values() if any(v != 0 for v in actions.values()))
        }
        
        for i, (stat, value) in enumerate(stats.items()):
            ttk.Label(q_table_info, text=f"{stat}:", font=('Helvetica', 9, 'bold')).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Label(q_table_info, text=str(value)).grid(row=i, column=1, sticky="w", padx=10, pady=2)
        
        # Show sample of Q-table
        ttk.Label(scrollable_frame, text="\nSample Q-table Entries:", font=('Helvetica', 10, 'bold')).pack(anchor="w", pady=(10,5))
        sample_frame = ttk.Frame(scrollable_frame)
        sample_frame.pack(fill=tk.X, padx=10)
        
        # Show up to 5 random states
        sample_states = list(agent.q_table.items())[:5]
        for i, (state, actions) in enumerate(sample_states):
            ttk.Label(sample_frame, text=f"State {i+1}:", font=('Helvetica', 9, 'bold')).grid(row=i*2, column=0, sticky="w", pady=(5,0))
            ttk.Label(sample_frame, text=str(actions)).grid(row=i*2+1, column=0, columnspan=2, sticky="w", padx=10)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Add close button
        ttk.Button(info_window, text="Close", command=info_window.destroy).pack(pady=10)
    
    def update_win_rate_graph(self):
        """Update the win rate graph with data for both players."""
        self.ax.clear()
        
        # Plot data for both players if we have any games
        if self.games_played > 0:
            x_data = range(1, len(self.x_wins) + 1)
            
            # Plot X player win rate
            self.ax.plot(x_data, self.x_wins, 'b-', label='Player X', alpha=0.7)
            
            # Plot O player win rate
            self.ax.plot(x_data, self.o_wins, 'r-', label='Player O', alpha=0.7)
            
            # Plot tie rate as a dotted line
            tie_rates = [1 - x - o for x, o in zip(self.x_wins, self.o_wins)]
            self.ax.plot(x_data, tie_rates, 'g--', label='Ties', alpha=0.5)
        
        self.ax.set_title('Win Rates Over Time')
        self.ax.set_xlabel('Games Played')
        self.ax.set_ylabel('Win Rate')
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Add player type labels
        x_type = self.player_x.get()
        o_type = self.player_o.get()
        self.ax.text(0.02, 0.98, f'X: {x_type}', transform=self.ax.transAxes, 
                    verticalalignment='top', color='blue')
        self.ax.text(0.02, 0.94, f'O: {o_type}', transform=self.ax.transAxes, 
                    verticalalignment='top', color='red')
        
        self.canvas.draw()
    
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
