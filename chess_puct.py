import random
import math
import chess
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ----------------------------
# Global parameters for network output size.
# ----------------------------
ACTION_SIZE = 4672  # Fixed action space size (as used in some chess engines)

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
# Neural Network for Chess (similar to AlphaZeroNet)
# ----------------------------
class AlphaZeroNet(nn.Module):
    def __init__(self):
        super(AlphaZeroNet, self).__init__()
        # The board is represented as 12 channels (pieces) x 8 x 8.
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Policy head
        self.conv_policy = nn.Conv2d(128, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * 8 * 8, ACTION_SIZE)
        
        # Value head
        self.conv_value = nn.Conv2d(128, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(8 * 8, 64)
        self.fc_value2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # x shape: (batch, 12, 8, 8)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Policy head
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(p.size(0), -1)
        p = self.fc_policy(p)
        p = F.softmax(p, dim=1)  # output probabilities for ACTION_SIZE moves
        
        # Value head
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value1(v))
        v = torch.tanh(self.fc_value2(v))  # value in [-1, 1]
        
        return p, v

# ----------------------------
# Helper: Convert ChessGame state to a tensor.
# ----------------------------
def board_to_tensor(game):
    # 12 channels: for white pieces (P,N,B,R,Q,K) and black pieces (p,n,b,r,q,k)
    tensor = torch.zeros((12, 8, 8), dtype=torch.float32)
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    }
    board = game.board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            symbol = piece.symbol()
            channel = piece_to_channel.get(symbol)
            if channel is not None:
                # Map chess square to tensor indices (row 0 at top)
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                tensor[channel, row, col] = 1.0
    return tensor.unsqueeze(0)  # shape: (1, 12, 8, 8)

# ----------------------------
# Mapping from a move to an index (a simple hash-based mapping)
# ----------------------------
def move_to_index(move):
    return abs(hash(move.uci())) % ACTION_SIZE

# ----------------------------
# Given a board state and the network, evaluate the state.
# Returns:
#   policy: a dict mapping each legal move to its prior probability.
#   value: a scalar value prediction.
# ----------------------------
def evaluate_state(state, network):
    tensor = board_to_tensor(state)
    device = next(network.parameters()).device
    tensor = tensor.to(device)
    with torch.no_grad():
        policy_logits, value = network(tensor)
    policy_logits = policy_logits.squeeze(0)  # shape: (ACTION_SIZE,)
    value = value.item()
    
    legal_moves = state.get_possible_moves()
    policy = {}
    probs = []
    moves_list = []
    for move in legal_moves:
        idx = move_to_index(move)
        prob = policy_logits[idx].item()
        probs.append(prob)
        moves_list.append(move)
    # Compute softmax only over legal moves
    max_prob = max(probs) if probs else 0
    exps = [math.exp(p - max_prob) for p in probs]
    sum_exps = sum(exps)
    softmax = [exp_val/sum_exps for exp_val in exps] if sum_exps > 0 else [1/len(probs)] * len(probs)
    for move, sm in zip(moves_list, softmax):
        policy[move] = sm
    return policy, value

# ----------------------------
# Modified MCTS Node class for PUCT
# ----------------------------
class MCTSNode:
    def __init__(self, state, parent, move, player=None, prior=0.0):
        self.state = state              # ChessGame state
        self.parent = parent            # Parent node
        self.move = move                # Move that led to this state
        self.player = player            # The player who made that move
        self.wins = 0.0               # Total reward
        self.visits = 0
        self.children = []
        self.untried_moves = state.get_possible_moves()
        self.prior = prior            # Prior probability from network
        self.expanded = False         # Whether this node has been expanded using network evaluation

    def add_child(self, move, state, player, prior):
        child_node = MCTSNode(state, self, move, player, prior)
        self.children.append(child_node)
        if move in self.untried_moves:
            self.untried_moves.remove(move)
        return child_node

    def best_child(self, c_param):
        best_score = -float('inf')
        best_child = None
        # PUCT: Q + U, where U = c_param * prior * sqrt(parent.visits) / (1 + child.visits)
        for child in self.children:
            Q = child.wins / child.visits if child.visits > 0 else 0
            U = c_param * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            score = Q + U
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def update(self, reward):
        self.visits += 1
        self.wins += reward

# ----------------------------
# MCTS with PUCT using network evaluations
# ----------------------------
def mcts_puct(root_state, network, itermax, c_param=1.4, verbose=False):
    root_node = MCTSNode(root_state, parent=None, move=None, player=None)
    root_player = root_state.current_player  # Perspective for reward
    for i in range(itermax):
        node = root_node
        state = deepcopy(root_state)
        path = [node]

        # Selection / Expansion / Evaluation
        while True:
            if state.is_terminal():
                reward = state.get_reward(root_player)
                break
            if not node.expanded:
                # Expand this leaf using network evaluation
                policy, value = evaluate_state(state, network)
                legal_moves = state.get_possible_moves()
                for move in legal_moves:
                    next_state = deepcopy(state)
                    next_state.make_move(move)
                    child_player = not state.current_player
                    node.add_child(move, next_state, child_player, prior=policy.get(move, 0))
                node.expanded = True
                reward = value  # Use network value as the simulation result
                break
            else:
                node = node.best_child(c_param)
                path.append(node)
                state.make_move(node.move)

        # Backpropagation along the visited path
        for n in path:
            # Flip reward sign for opponent moves
            if n.player is None:
                n.update(reward)
            elif n.player == root_player:
                n.update(reward)
            else:
                n.update(-reward)

        if verbose:
            ucb_values = {}
            for child in root_node.children:
                Q = child.wins / child.visits if child.visits > 0 else 0
                U = c_param * child.prior * math.sqrt(root_node.visits) / (1 + child.visits)
                ucb_values[child.move.uci()] = Q + U
            print(f"Iteration {i}: UCB values at root: {ucb_values}")
    return root_node

# ----------------------------
# Helper: Convert target policy dict to a fixed-size vector for training.
# ----------------------------
def policy_dict_to_vector(policy):
    vec = torch.zeros(ACTION_SIZE, dtype=torch.float32)
    for move, prob in policy.items():
        idx = move_to_index(move)
        vec[idx] = prob
    return vec

# ----------------------------
# Self-play: Let the AI play a full game against itself.
# Also collect training examples: (state_tensor, target_policy, reward)
# ----------------------------
def self_play_game(network, mcts_iterations, c_param):
    training_examples = []
    game = ChessGame()
    states, mcts_policies, current_players = [], [], []
    
    while not game.is_terminal():
        state_tensor = board_to_tensor(game)
        root_node = mcts_puct(game, network, mcts_iterations, c_param)
        # Build a target policy from visit counts at the root.
        visit_counts = {}
        for child in root_node.children:
            visit_counts[child.move] = child.visits
        total_visits = sum(visit_counts.values())
        target_policy = {}
        for move, count in visit_counts.items():
            target_policy[move] = count / total_visits if total_visits > 0 else 0
        
        states.append(state_tensor)
        mcts_policies.append(target_policy)
        current_players.append(game.current_player)
        
        # Choose move with most visits.
        best_child = max(root_node.children, key=lambda c: c.visits)
        game.make_move(best_child.move)
    
    # Determine game outcome.
    winner = game.get_winner()
    # Assign rewards from the perspective of the player at each state.
    for state_tensor, policy, player in zip(states, mcts_policies, current_players):
        if winner is None:
            reward = 0
        else:
            reward = 1 if player == winner else -1
        training_examples.append((state_tensor, policy_dict_to_vector(policy), reward))
    return training_examples

# ----------------------------
# Training loop: update the network using self-play examples.
# ----------------------------
def train_network(network, optimizer, training_examples, epochs=1, batch_size=16, device='cpu'):
    network.train()
    total_loss = 0.0
    # Shuffle the training examples.
    random.shuffle(training_examples)
    for epoch in range(epochs):
        for i in range(0, len(training_examples), batch_size):
            batch = training_examples[i:i+batch_size]
            # Prepare batch tensors.
            state_batch = torch.cat([example[0] for example in batch], dim=0).to(device)
            target_policy_batch = torch.stack([example[1] for example in batch]).to(device)
            target_value_batch = torch.tensor([example[2] for example in batch], dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            pred_policy, pred_value = network(state_batch)
            # Policy loss: cross-entropy (using log probabilities)
            eps = 1e-8
            loss_policy = -torch.mean(torch.sum(target_policy_batch * torch.log(pred_policy + eps), dim=1))
            # Value loss: mean squared error
            loss_value = torch.mean((target_value_batch - pred_value.squeeze()) ** 2)
            loss = loss_policy + loss_value
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss

# ----------------------------
# Main: training and inference (AI self-play)
# ----------------------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = AlphaZeroNet().to(device)
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    # Parameters for self-play and MCTS.
    mcts_iterations = 200   # Increase for stronger play
    c_param = 1.4
    num_self_play_games = 5  # For demonstration; typically many more games are needed.
    training_epochs = 3

    # --- Self-Play Training ---
    print("Starting self-play training...")
    training_data = []
    for game_num in range(num_self_play_games):
        print(f"Self-play game {game_num+1}/{num_self_play_games}...")
        examples = self_play_game(network, mcts_iterations, c_param)
        training_data.extend(examples)

    loss = train_network(network, optimizer, training_data, epochs=training_epochs, device=device)
    print(f"Training completed. Total loss: {loss:.4f}\n")

    # --- Inference: AI plays against itself ---
    print("Starting inference: AI playing against itself...\n")
    game = ChessGame()
    move_num = 1
    while not game.is_terminal():
        print(f"Move {move_num}:")
        game.print_board()
        root_node = mcts_puct(game, network, mcts_iterations, c_param, verbose=False)
        best_child = max(root_node.children, key=lambda c: c.visits)
        chosen_move = best_child.move
        print(f"AI selects move: {chosen_move.uci()}\n")
        game.make_move(chosen_move)
        move_num += 1

    print("Final board:")
    game.print_board()
    winner = game.get_winner()
    if winner is None:
        print("The game ended in a draw.")
    else:
        player_str = "White" if winner == chess.WHITE else "Black"
        print(f"The winner is: {player_str}")
