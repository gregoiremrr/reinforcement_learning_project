import random
import math
from copy import deepcopy

# ----------------------------
# Tic Tac Toe game implementation
# ----------------------------
class TicTacToe:
    def __init__(self):
        # Represent the board as a list of 9 cells (None means empty)
        self.board = [None] * 9  
        self.current_player = 'X'  # 'X' always starts

    def get_possible_moves(self):
        # Return list of indices (0-8) where a move can be made
        return [i for i, cell in enumerate(self.board) if cell is None]

    def make_move(self, move):
        if self.board[move] is None:
            self.board[move] = self.current_player
            # Switch the player for the next move
            self.current_player = 'O' if self.current_player == 'X' else 'X'
        else:
            raise Exception("Invalid move: cell already taken.")

    def is_terminal(self):
        # The game ends if there's a winner or the board is full
        return self.get_winner() is not None or None not in self.board

    def get_winner(self):
        # Check all winning combinations: rows, columns, and diagonals
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
            (0, 4, 8), (2, 4, 6)              # diagonals
        ]
        for (a, b, c) in lines:
            if self.board[a] is not None and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return None

    def get_reward(self, player):
        # Reward from the perspective of 'player'
        # +1 if player wins, -1 if loses, 0 for draw.
        winner = self.get_winner()
        if winner == player:
            return 1
        elif winner is None and None not in self.board:
            return 0  # Draw
        elif winner is None:
            return 0  # Not terminal (should not happen in rollout)
        else:
            return -1

    def print_board(self):
        # Print the board in a 3x3 format
        for i in range(0, 9, 3):
            row = [self.board[j] if self.board[j] is not None else '.' for j in range(i, i+3)]
            print(' '.join(row))
        print()

# ----------------------------
# MCTS Node class
# ----------------------------
class MCTSNode:
    def __init__(self, state, parent, move, player=None):
        self.state = state              # Game state at this node
        self.parent = parent            # Parent node
        self.move = move                # Move that led to this state (None for the root)
        self.player = player            # The player who made that move (None for the root)
        self.wins = 0                   # Total reward (from this node’s perspective)
        self.visits = 0                 # Number of times this node was visited
        self.children = []              # List of child nodes
        self.untried_moves = state.get_possible_moves()  # Moves not yet tried from this state

    def add_child(self, move, state, player):
        """Add a new child node for the move, recording the player who made the move."""
        child_node = MCTSNode(state, self, move, player)
        self.untried_moves.remove(move)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        self.visits += 1
        self.wins += reward

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.wins / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

# ----------------------------
# MCTS algorithm function
# ----------------------------
def mcts(root_state, itermax, verbose=False):
    # Create the root node (no move made to get here, so player is None)
    root_node = MCTSNode(root_state, parent=None, move=None, player=None)
    root_player = root_state.current_player  # This is the AI's perspective

    c_param = 1.4  # Exploration parameter for UCB

    for i in range(itermax):
        node = root_node
        state = deepcopy(root_state)
        path = []  # To record the moves along this iteration

        # --- Selection ---
        while node.untried_moves == [] and node.children != [] and not state.is_terminal():
            node = node.best_child()
            if node.move is not None:
                path.append(node.move)
            state.make_move(node.move)

        # --- Expansion ---
        if not state.is_terminal() and node.untried_moves:
            move = random.choice(node.untried_moves)
            # Before making the move, state.current_player is the player who is about to move.
            state.make_move(move)
            # The move was made by the opposite of the new state.current_player.
            child_player = 'X' if state.current_player == 'O' else 'O'
            node = node.add_child(move, deepcopy(state), child_player)
            path.append(move)

        # --- Simulation (Rollout) with immediate terminal check ---
        while True:
            if state.is_terminal():
                break
            possible_moves = state.get_possible_moves()
            if not possible_moves:
                break
            move = random.choice(possible_moves)
            state.make_move(move)
            path.append(move)
            if state.is_terminal():
                break

        # Get simulation reward from the perspective of the root player
        sim_reward = state.get_reward(root_player)
        if verbose:
            print(f"Iteration {i}: path = {path} with reward = {sim_reward}")

        # --- Backpropagation ---
        current_node = node
        # Walk back up the tree
        while current_node is not None:
            if current_node.player is None:
                # Root node – update directly (though its value isn’t used for decision)
                current_node.update(sim_reward)
            elif current_node.player == root_player:
                # Node made by the AI; update with sim_reward (from AI perspective)
                current_node.update(sim_reward)
            else:
                # Node made by the opponent; update with -sim_reward
                current_node.update(-sim_reward)
            current_node = current_node.parent

        # --- Verbose UCB printout for selected moves at the root ---
        if verbose:
            moves_to_track = [1, 3, 6, 7, 8]
            ucb_values = {}
            for move in moves_to_track:
                child = next((child for child in root_node.children if child.move == move), None)
                if child is not None:
                    ucb = (child.wins / child.visits) + c_param * math.sqrt(math.log(root_node.visits) / child.visits)
                    ucb_values[move] = ucb
                else:
                    ucb_values[move] = "Not explored"
            print(f"Iteration {i}: UCB values at root: {ucb_values}")

    return root_node

def best_move(root_node, version="most_visits"):
    if version=="most_visits":
        # Select the move that was visited the most
        best_child_node = max(root_node.children, key=lambda c: c.visits)
        return best_child_node.move
    if version=="best_average":
        # Select the move with the best average reward (wins/visits)
        best_child_node = max(root_node.children, key=lambda c: c.wins / c.visits)
        return best_child_node.move
    else:
        raise ValueError("Invalid version parameter. Choose 'most_visits' or 'best_average'.")

# ----------------------------
# Play an entire game using MCTS for both players
# ----------------------------
if __name__ == '__main__2':
    game = TicTacToe()
    print("Starting Tic Tac Toe game with MCTS for both players:\n")
    
    while not game.is_terminal():
        print("Current board:")
        game.print_board()
        # Run MCTS for a set number of iterations for the current game state
        iterations = 1000
        root_node = mcts(game, iterations, verbose=False)
        move = best_move(root_node)
        print(f"Player {game.current_player} selects move: {move}\n")
        game.make_move(move)
    
    print("Final board:")
    game.print_board()
    winner = game.get_winner()
    if winner:
        print(f"The winner is: {winner}")
    else:
        print("The game ended in a draw.")

if __name__ == '__main__':
    game = TicTacToe()
    # Choose to play first or second
    choice = input("Do you want to play first or second? (Enter 'f' for first, 's' for second): ").strip().lower()
    if choice == 'f':
        human_player = 'X'
    else:
        human_player = 'O'
    print(f"You are playing as {human_player}.\n")

    # Explain board indexing to the user
    print("Board positions are numbered as follows:")
    print("0 | 1 | 2")
    print("---------")
    print("3 | 4 | 5")
    print("---------")
    print("6 | 7 | 8\n")

    while not game.is_terminal():
        print("Current board:")
        game.print_board()

        if game.current_player == human_player:
            # Human move
            valid_move = False
            while not valid_move:
                try:
                    move = int(input(f"Your turn ({human_player}). Enter your move (0-8): "))
                    if move in game.get_possible_moves():
                        valid_move = True
                    else:
                        print("Invalid move! Try again.")
                except ValueError:
                    print("Please enter a valid integer between 0 and 8.")
            game.make_move(move)
        else:
            # AI move using MCTS
            print(f"AI ({game.current_player}) is thinking...")
            iterations = 100
            root_node = mcts(game, iterations, verbose=False)
            move = best_move(root_node)
            print(f"AI selects move: {move}\n")
            game.make_move(move)

    print("Final board:")
    game.print_board()
    winner = game.get_winner()
    if winner:
        if winner == human_player:
            print("Congratulations! You won!")
        else:
            print("AI wins! Better luck next time.")
    else:
        print("The game ended in a draw.")