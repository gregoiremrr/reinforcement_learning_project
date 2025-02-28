import random
import math
import chess
from copy import deepcopy

# ----------------------------
# Chess game wrapper class
# ----------------------------
class ChessGame:
    def __init__(self):
        self.board = chess.Board()

    @property
    def current_player(self):
        # chess.Board.turn returns True for White, False for Black
        return self.board.turn

    def get_possible_moves(self):
        return list(self.board.legal_moves)

    def make_move(self, move):
        self.board.push(move)

    def is_terminal(self):
        return self.board.is_game_over()

    def get_winner(self):
        if not self.is_terminal():
            return None
        # board.result() returns "1-0", "0-1", or "1/2-1/2"
        result = self.board.result()
        if result == "1-0":
            return chess.WHITE
        elif result == "0-1":
            return chess.BLACK
        else:
            return None  # Draw

    def get_reward(self, player):
        winner = self.get_winner()
        if winner == player:
            return 1
        elif winner is None:
            return 0
        else:
            return -1

    def print_board(self):
        print(self.board)
        print()

    def __deepcopy__(self, memo):
        new_game = ChessGame()
        new_game.board = self.board.copy()
        return new_game

# ----------------------------
# MCTS Node class (same structure as before)
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
        self.untried_moves = state.get_possible_moves()  # Moves not yet tried

    def add_child(self, move, state, player):
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
# MCTS algorithm function (minimal modifications from the Tic Tac Toe version)
# ----------------------------
def mcts(root_state, itermax, verbose=False):
    root_node = MCTSNode(root_state, parent=None, move=None, player=None)
    root_player = root_state.current_player  # AI's perspective
    c_param = 1.4

    for i in range(itermax):
        node = root_node
        state = deepcopy(root_state)
        path = []  # Record moves along the path

        # --- Selection ---
        while node.untried_moves == [] and node.children != [] and not state.is_terminal():
            node = node.best_child()
            if node.move is not None:
                path.append(node.move)
            state.make_move(node.move)

        # --- Expansion ---
        if not state.is_terminal() and node.untried_moves:
            move = random.choice(node.untried_moves)
            state.make_move(move)
            # After move, state.current_player is the one who is about to move,
            # so the move was made by the opposite.
            child_player = not state.current_player
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

        sim_reward = state.get_reward(root_player)
        if verbose:
            # Convert moves to UCI strings for readability.
            path_str = [move.uci() for move in path]
            print(f"Iteration {i}: path = {path_str} with reward = {sim_reward}")

        # --- Backpropagation ---
        current_node = node
        reward = sim_reward
        while current_node is not None:
            if current_node.player is None:
                current_node.update(sim_reward)
            elif current_node.player == root_player:
                current_node.update(sim_reward)
            else:
                current_node.update(-sim_reward)
            current_node = current_node.parent

        # --- Verbose UCB printout for root's children ---
        if verbose:
            ucb_values = {}
            for child in root_node.children:
                move_uci = child.move.uci()
                ucb = (child.wins / child.visits) + c_param * math.sqrt(math.log(root_node.visits) / child.visits)
                ucb_values[move_uci] = ucb
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
# Main game loop: Human vs AI in Chess
# ----------------------------
if __name__ == '__main__':
    game = ChessGame()
    print("Starting Chess game with MCTS for both players:\n")
    
    while not game.is_terminal():
        print("Current board:")
        game.print_board()
        print("AI is thinking...")
        # Run MCTS for a set number of iterations for the current game state
        iterations = 500
        root_node = mcts(game, iterations, verbose=False)
        ai_move = best_move(root_node)
        print(f"Player {game.current_player} selects move: {ai_move}\n")
        game.make_move(ai_move)
    
    print("Final board:")
    game.print_board()
    winner = game.get_winner()
    if winner:
        print(f"The winner is: {winner}")
    else:
        print("The game ended in a draw.")

if __name__ == '__main__2':
    game = ChessGame()
    choice = input("Do you want to play as White or Black? (Enter 'w' for White, 'b' for Black): ").strip().lower()
    if choice == 'w':
        human_player = chess.WHITE
    else:
        human_player = chess.BLACK

    print(f"You are playing as {'White' if human_player == chess.WHITE else 'Black'}.\n")

    while not game.is_terminal():
        game.print_board()
        if game.current_player == human_player:
            valid_move = False
            while not valid_move:
                user_input = input("Your move (in UCI format, e.g., e2e4): ").strip()
                try:
                    move = chess.Move.from_uci(user_input)
                    if move in game.get_possible_moves():
                        valid_move = True
                    else:
                        print("Invalid move (not legal). Try again.")
                except Exception:
                    print("Invalid move format. Please try again.")
            game.make_move(move)
        else:
            print("AI is thinking...")
            iterations = 500  # You can adjust the number of iterations
            root_node = mcts(game, iterations, verbose=False)
            ai_move = best_move(root_node)
            print(f"AI selects move: {ai_move.uci()}\n")
            game.make_move(ai_move)

    game.print_board()
    winner = game.get_winner()
    if winner is None:
        print("Game ended in a draw.")
    elif winner == human_player:
        print("Congratulations! You win!")
    else:
        print("AI wins! Better luck next time!")
