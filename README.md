# Tic-Tac-Toe AI Arena

A Python implementation of Tic-Tac-Toe featuring multiple AI agents, including a learning AI that uses Q-Learning to improve its gameplay over time. The project includes a graphical user interface built with Tkinter and real-time performance visualization using Matplotlib.

## Project Structure

The project consists of several Python modules, each handling different aspects of the game:

### `src/main.py`
The main entry point of the application that implements the graphical user interface (TicTacToeArena). Features include:
- AI agent selection for both players (X and O)
- Configurable match settings (number of games, move delay)
- Real-time game board visualization
- Performance statistics and graphs
- Match control (start/stop functionality)

### `src/game.py`
Implements the core game logic (TicTacToeGame class) with features such as:
- Board state management using NumPy arrays
- Move validation and execution
- Win condition checking
- Game state tracking
- Board rendering

### `src/agents.py`
Contains different AI agent implementations:
- `Agent`: Base class for all agents
- `HumanAgent`: Allows human players to input moves via console
- `RandomAgent`: Makes random valid moves
- `MinimaxAgent`: Implements a perfect player using the minimax algorithm with alpha-beta pruning

### `src/rl_agent.py`
Implements a Q-Learning agent (QLearningAgent) that learns to play through experience:
- Epsilon-greedy action selection
- Q-value table management
- Reward calculation based on game state
- Persistent learning through Q-table saving/loading
- Adaptive exploration rate

## Training Process

The Learning AI uses Q-Learning to improve its gameplay through experience. The training process involves:

1. **State Representation**
   - The game board is converted to a string representation
   - Each state-action pair has an associated Q-value

2. **Action Selection**
   - Uses epsilon-greedy strategy for exploration vs exploitation
   - Epsilon decays over time (from 0.9 to 0.05)
   - Balances exploring new strategies and exploiting known good moves

3. **Reward System**
   - Win: +5.0
   - Tie: +1.0
   - Loss: -3.0
   - Blocking opponent win: +2.0
   - Creating winning opportunity: +1.5
   - Taking center: +0.8
   - Taking corner: +0.5
   - Missing strategic positions: -0.1 to -0.2

4. **Learning Updates**
   - Uses Q-Learning update rule: Q(s,a) = Q(s,a) + lr * (r + γ * max(Q(s')) - Q(s,a))
   - Learning rate (lr) = 0.4
   - Discount factor (γ) = 0.95

5. **Evaluation**
   - Performance tracked against both Random and Perfect (Minimax) players
   - Metrics include win rate, tie rate, and non-loss rate
   - Real-time visualization of learning progress
   - Historical performance tracking

## Key Features

1. **Multiple AI Types**
   - Random AI: Makes random valid moves
   - Perfect AI: Uses minimax algorithm for optimal play
   - Learning AI: Improves through Q-Learning

2. **Interactive GUI**
   - Real-time game board display
   - Performance statistics tracking
   - Configurable match settings

3. **Performance Analysis**
   - Win/loss/tie statistics
   - Learning progress visualization
   - Historical performance tracking

4. **Reinforcement Learning**
   - Q-Learning implementation
   - Persistent learning across sessions
   - Configurable learning parameters
   - Strategic reward system

## Dependencies

The project requires the following Python packages:
- numpy
- tkinter
- matplotlib

## Usage

1. Run the main script to start the application:
   ```
   python src/main.py
   ```

2. Select AI players for both X and O positions
3. Configure match settings (number of games, move delay)
4. Click "Start Match" to begin the game

The Learning AI will improve its performance over time as it plays more games. Its learned strategies are automatically saved to `q_table.json` and loaded in subsequent runs.

## Training Tips

1. **Initial Training**
   - Start by training against the Random AI to learn basic strategies
   - Run at least 1000 games to build up initial knowledge

2. **Advanced Training**
   - Switch to training against the Perfect AI to refine strategies
   - Expect lower win rates but focus on increasing tie rate
   - Run multiple sessions to accumulate experience

3. **Monitoring Progress**
   - Watch the win rate against Random AI (should reach >80%)
   - Monitor tie rate against Perfect AI (should increase over time)
   - Check if strategic positions (center, corners) are prioritized 