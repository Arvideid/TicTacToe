// Game state and configuration
let isMatchRunning = false;
let currentPlayer = 'X';
let matchStats = { x_wins: 0, o_wins: 0, ties: 0 };
let winRateChart, nonLossRateChart;
let gameTimeout = null;
let gamesRemaining = 0;

// API configuration
const API_BASE_URL = 'http://localhost:5000/api';

// DOM Elements
const cells = document.querySelectorAll('.cell');
const statusText = document.getElementById('status-text');
const startMatchBtn = document.getElementById('start-match');
const stopMatchBtn = document.getElementById('stop-match');
const playerXSelect = document.getElementById('player-x');
const playerOSelect = document.getElementById('player-o');
const numGamesInput = document.getElementById('num-games');
const moveDelayInput = document.getElementById('move-delay');

// Tab handling
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabButtons.forEach(button => {
    button.addEventListener('click', () => {
        // Remove active class from all buttons and contents
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(content => content.classList.remove('active'));
        
        // Add active class to clicked button and corresponding content
        button.classList.add('active');
        const tabId = button.dataset.tab;
        document.getElementById(tabId).classList.add('active');
    });
});

// Initialize charts
function initializeCharts() {
    const winRateCtx = document.getElementById('win-rate-chart').getContext('2d');
    const nonLossRateCtx = document.getElementById('non-loss-rate-chart').getContext('2d');

    winRateChart = new Chart(winRateCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'vs Random',
                    data: [],
                    borderColor: '#4a90e2',
                    tension: 0.1
                },
                {
                    label: 'vs Perfect',
                    data: [],
                    borderColor: '#e74c3c',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Win Rate vs Episodes'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });

    nonLossRateChart = new Chart(nonLossRateCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'vs Random',
                    data: [],
                    borderColor: '#2ecc71',
                    tension: 0.1
                },
                {
                    label: 'vs Perfect',
                    data: [],
                    borderColor: '#f1c40f',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Non-Loss Rate vs Episodes'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

// Update game board UI
function updateBoard(board) {
    cells.forEach((cell, index) => {
        const row = Math.floor(index / 3);
        const col = index % 3;
        const value = board[row][col];
        
        cell.textContent = value === 1 ? 'X' : value === -1 ? 'O' : '';
        cell.className = 'cell' + (value === 1 ? ' x' : value === -1 ? ' o' : '');
    });
}

// API calls
async function makeMove(row, col, agent) {
    try {
        const response = await fetch(`${API_BASE_URL}/move`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                row: row,
                col: col,
                agent: agent
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        // Update board and game state
        updateBoard(data.board);
        currentPlayer = data.current_player;

        // Update performance charts if stats are available
        if (data.episodes && data.stats) {
            updateCharts(data.episodes, data.stats);
        }

        return data;
    } catch (error) {
        console.error('Error making move:', error);
        statusText.textContent = `Error: ${error.message}`;
        throw error;
    }
}

async function resetGame() {
    try {
        const response = await fetch(`${API_BASE_URL}/reset`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }
        return data;
    } catch (error) {
        console.error('Error resetting game:', error);
        statusText.textContent = `Error: ${error.message}`;
        throw error;
    }
}

// Update match statistics
function updateStats() {
    document.getElementById('x-wins').textContent = matchStats.x_wins;
    document.getElementById('o-wins').textContent = matchStats.o_wins;
    document.getElementById('ties').textContent = matchStats.ties;
}

// Update performance charts
function updateCharts(episodes, stats) {
    winRateChart.data.labels = episodes;
    nonLossRateChart.data.labels = episodes;

    // Update win rate data
    winRateChart.data.datasets[0].data = stats.vs_random.wins.map((w, i) => 
        w / (w + stats.vs_random.ties[i] + stats.vs_random.losses[i]));
    winRateChart.data.datasets[1].data = stats.vs_perfect.wins.map((w, i) => 
        w / (w + stats.vs_perfect.ties[i] + stats.vs_perfect.losses[i]));

    // Update non-loss rate data
    nonLossRateChart.data.datasets[0].data = stats.vs_random.wins.map((w, i) => 
        (w + stats.vs_random.ties[i]) / (w + stats.vs_random.ties[i] + stats.vs_random.losses[i]));
    nonLossRateChart.data.datasets[1].data = stats.vs_perfect.wins.map((w, i) => 
        (w + stats.vs_perfect.ties[i]) / (w + stats.vs_perfect.ties[i] + stats.vs_perfect.losses[i]));

    winRateChart.update();
    nonLossRateChart.update();
}

// Add game result to history
function addToHistory(result) {
    const historyLog = document.getElementById('history-log');
    const entry = document.createElement('div');
    entry.className = 'history-entry';
    entry.textContent = result;
    historyLog.insertBefore(entry, historyLog.firstChild);
}

// Make AI move if it's AI's turn
async function makeAIMove() {
    if (!isMatchRunning) return;
    
    const currentAgent = currentPlayer === 'X' ? playerXSelect.value : playerOSelect.value;
    if (currentAgent === 'human') return;

    try {
        const data = await makeMove(null, null, currentAgent);
        updateBoard(data.board);
        currentPlayer = data.current_player;
        
        if (data.game_over) {
            handleGameOver(data.winner);
            if (gamesRemaining > 0) {
                // Start next game after delay
                gameTimeout = setTimeout(startNextGame, parseInt(moveDelayInput.value));
            }
        } else {
            statusText.textContent = `Current player: ${currentPlayer}`;
            // If next player is also AI, make their move after delay
            const nextAgent = currentPlayer === 'X' ? playerXSelect.value : playerOSelect.value;
            if (nextAgent !== 'human') {
                gameTimeout = setTimeout(makeAIMove, parseInt(moveDelayInput.value));
            }
        }
    } catch (error) {
        stopMatch();
    }
}

// Start next game in the match
async function startNextGame() {
    if (!isMatchRunning || gamesRemaining <= 0) {
        stopMatch();
        return;
    }

    gamesRemaining--;
    statusText.textContent = `Games remaining: ${gamesRemaining}`;
    
    try {
        const data = await resetGame();
        updateBoard(data.board);
        currentPlayer = data.current_player;
        
        // Start AI moves if first player is AI
        const firstAgent = playerXSelect.value;
        if (firstAgent !== 'human') {
            gameTimeout = setTimeout(makeAIMove, parseInt(moveDelayInput.value));
        }
    } catch (error) {
        stopMatch();
    }
}

// Handle cell click for human moves
async function handleCellClick(row, col) {
    if (!isMatchRunning) return;
    if (currentPlayer === 'X' && playerXSelect.value !== 'human') return;
    if (currentPlayer === 'O' && playerOSelect.value !== 'human') return;
    
    try {
        const data = await makeMove(row, col, 
            currentPlayer === 'X' ? playerOSelect.value : playerXSelect.value);
        
        updateBoard(data.board);
        currentPlayer = data.current_player;
        
        if (data.game_over) {
            handleGameOver(data.winner);
            if (gamesRemaining > 0) {
                gameTimeout = setTimeout(startNextGame, parseInt(moveDelayInput.value));
            }
        } else {
            statusText.textContent = `Current player: ${currentPlayer}`;
            // If next player is AI, make their move after delay
            const nextAgent = currentPlayer === 'X' ? playerXSelect.value : playerOSelect.value;
            if (nextAgent !== 'human') {
                gameTimeout = setTimeout(makeAIMove, parseInt(moveDelayInput.value));
            }
        }
    } catch (error) {
        stopMatch();
    }
}

// Handle game over
function handleGameOver(winner) {
    if (winner === 1) {
        matchStats.x_wins++;
        statusText.textContent = 'Game Over - X wins!';
    } else if (winner === -1) {
        matchStats.o_wins++;
        statusText.textContent = 'Game Over - O wins!';
    } else {
        matchStats.ties++;
        statusText.textContent = 'Game Over - Tie!';
    }
    updateStats();
}

// Start new match
async function startMatch() {
    try {
        isMatchRunning = true;
        startMatchBtn.disabled = true;
        stopMatchBtn.disabled = false;
        playerXSelect.disabled = true;
        playerOSelect.disabled = true;
        
        // Reset match stats
        matchStats = { x_wins: 0, o_wins: 0, ties: 0 };
        updateStats();
        
        // Set number of games
        gamesRemaining = parseInt(numGamesInput.value);
        
        const data = await resetGame();
        updateBoard(data.board);
        currentPlayer = data.current_player;
        statusText.textContent = `Games remaining: ${gamesRemaining}`;
        
        // Start AI moves if first player is AI
        const firstAgent = playerXSelect.value;
        if (firstAgent !== 'human') {
            gameTimeout = setTimeout(makeAIMove, parseInt(moveDelayInput.value));
        }
    } catch (error) {
        stopMatch();
    }
}

// Stop current match
function stopMatch() {
    isMatchRunning = false;
    startMatchBtn.disabled = false;
    stopMatchBtn.disabled = true;
    playerXSelect.disabled = false;
    playerOSelect.disabled = false;
    if (gameTimeout) {
        clearTimeout(gameTimeout);
        gameTimeout = null;
    }
    statusText.textContent = 'Match stopped';
    gamesRemaining = 0;
}

// Initialize game
function initGame() {
    // Add event listeners
    cells.forEach(cell => {
        const row = parseInt(cell.dataset.row);
        const col = parseInt(cell.dataset.col);
        cell.addEventListener('click', () => handleCellClick(row, col));
    });

    startMatchBtn.addEventListener('click', startMatch);
    stopMatchBtn.addEventListener('click', stopMatch);

    // Initialize charts
    initializeCharts();
}

// Start the game when the page loads
document.addEventListener('DOMContentLoaded', initGame); 