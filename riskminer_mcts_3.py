import math
import copy
import random
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datetime import datetime

###############################################################################
# 1. Define Operators, Operands, and Special Tokens
###############################################################################
OPERATORS = {
    'Sign': 1,
    'Abs': 1,
    'Log': 1,
    #'CSRank': 1,   # Placeholder; not implemented below.
    '+': 2,
    '-': 2,
    '*': 2,
    '/': 2,
    'Greater': 2,
    'Less': 2,
    #'Ref': 2,
    # Additional operators (Rank, Skew, etc.) can be added later.
}

OPERANDS = [
    "open", "high", "close", "low", "volume", "vwap",
    1, 5, 10, 20, 30, 40, 50,
    -30.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0
]

BEG = "BEG"
END = "END"

###############################################################################
# 2. Helper Functions for RPN Evaluation and Operator Application
###############################################################################
def apply_operator(op, x, y=None):
    """
    Apply an operator to the given pandas Series.
    x and y are assumed to be pandas Series.
    """
    if op == "Sign":
        return x.apply(lambda v: 1.0 if v > 0 else 0.0)
    elif op == "Abs":
        return x.abs()
    elif op == "Log":
        # Clip to avoid log(0) or negatives.
        return np.log(x.clip(lower=1e-8))
    elif op == "+":
        return x + y
    elif op == "-":
        return x - y
    elif op == "*":
        return x * y
    elif op == "/":
        # Replace zeros in denominator with a small number.
        return x / y.replace(0, 1e-8)
    elif op == "Greater":
        return (x > y).astype(float)
    elif op == "Less":
        return (x < y).astype(float)
    elif op == "Ref":
        # For Ref(x, t): shift x by t periods. Assume t is constant.
        lag = int(y.iloc[0]) if isinstance(y, pd.Series) else int(y)
        return x.shift(lag)
    else:
        raise NotImplementedError(f"Operator {op} not implemented in compute_ic.")

def evaluate_rpn(rpn_tokens, df):
    """
    Evaluate an RPN expression (list of tokens) on the DataFrame df.
    For demonstration, we use the "^FTSE" ticker.
    
    For each token:
      - If token is one of the price/volume features, map it to the corresponding column.
      - If token is a constant (or numeric string), create a constant Series.
      - If token is an operator, apply it to the required number of Series from the stack.
    """
    stack = []
    for token in rpn_tokens:
        if token in ["open", "high", "close", "low", "volume", "vwap"]:
            col_name = f"^FTSE_{token.capitalize()}"
            if col_name not in df.columns:
                raise ValueError(f"Column {col_name} not found in data.")
            stack.append(df[col_name])
        elif token in OPERATORS:
            arity = OPERATORS[token]
            if len(stack) < arity:
                raise ValueError(f"Insufficient operands for operator {token}")
            if arity == 1:
                x = stack.pop()
                result = apply_operator(token, x)
            elif arity == 2:
                y = stack.pop()  # second operand
                x = stack.pop()  # first operand
                result = apply_operator(token, x, y)
            else:
                raise NotImplementedError("Operators with arity > 2 not supported")
            stack.append(result)
        else:
            # Try to convert token to a constant.
            try:
                const_value = float(token)
                # Create a constant Series with the same index as df.
                stack.append(pd.Series(const_value, index=df.index))
            except ValueError:
                raise ValueError(f"Unknown token: {token}")
    if len(stack) != 1:
        raise ValueError("Invalid RPN expression: final stack size != 1")
    return stack[0]

###############################################################################
# 3. Compute IC from RPN Expression
###############################################################################
def compute_ic(rpn_expression):
    """
    Compute the Information Coefficient (IC) of an alpha.
    
    Steps:
      1. Load market data from "ftse100_data_wide.csv".
      2. For demonstration, use the "^FTSE" ticker.
      3. Evaluate the RPN expression (after removing BEG and END) on the data
         to produce a Series of alpha scores z_t.
      4. Compute next-day returns r_{t+1} from "^FTSE_Close".
      5. Align the two Series (dropping missing values) and compute:
         IC = Cov(z_t, r_{t+1}) / (std(z_t)*std(r_{t+1})).
    """
    # Load market data (assumes CSV file exists in the working directory)
    df = pd.read_csv("ftse100_data_wide.csv", index_col=0, parse_dates=True, low_memory=False)
    
    # Remove BEG and END tokens from the expression.
    tokens = [t for t in rpn_expression if t not in (BEG, END)]
    
    # Evaluate the RPN expression to get alpha scores (z_t).
    try:
        z = evaluate_rpn(tokens, df)
    except Exception as e:
        print(f"Error evaluating RPN: {e}")
        return 0.0
    
    # Compute future returns r_{t+1} from "^FTSE_Close".
    if "^FTSE_Close" not in df.columns:
        raise ValueError("Column '^FTSE_Close' not found in data.")
    close = df["^FTSE_Close"]
    r = close.pct_change(fill_method=None).shift(-1)
    
    # Align the two Series on their index.
    combined = pd.concat([z, r], axis=1, join="inner").dropna()
    if combined.empty:
        return 0.0
    z_aligned = combined.iloc[:, 0]
    r_aligned = combined.iloc[:, 1]
    
    # Compute covariance and standard deviations.
    cov = np.cov(z_aligned, r_aligned, ddof=1)[0, 1]
    std_z = np.std(z_aligned, ddof=1)
    std_r = np.std(r_aligned, ddof=1)
    if std_z == 0 or std_r == 0:
        return 0.0
    ic = cov / (std_z * std_r)
    return ic

def compute_mutic(rpn_expr_a, rpn_expr_b):
    """
    Compute the mutual IC (Pearson correlation) between two alpha expressions.
    
    Each expression (a list of tokens) is evaluated on the same market data 
    (from "ftse100_data_wide.csv") to produce a Series of alpha scores.
    The Pearson correlation between these series is returned.
    """
    
    # Load market data
    df = pd.read_csv("ftse100_data_wide.csv", index_col=0, parse_dates=True, low_memory=False)
    
    # Remove BEG and END tokens
    tokens_a = [t for t in rpn_expr_a if t not in (BEG, END)]
    tokens_b = [t for t in rpn_expr_b if t not in (BEG, END)]
    
    try:
        # Evaluate each RPN expression; reuse the evaluate_rpn function defined earlier.
        z_a = evaluate_rpn(tokens_a, df)
        z_b = evaluate_rpn(tokens_b, df)
    except Exception as e:
        print(f"Error evaluating RPN in compute_mutic: {e}")
        return 0.0
    
    # Align the two Series on their index, dropping missing values.
    combined = pd.concat([z_a, z_b], axis=1, join="inner").dropna()
    if combined.empty:
        return 0.0
    z_a_aligned = combined.iloc[:, 0]
    z_b_aligned = combined.iloc[:, 1]
    
    # Compute Pearson correlation.
    corr_matrix = np.corrcoef(z_a_aligned, z_b_aligned)
    corr = corr_matrix[0, 1]
    return corr

def update_composite_alpha(alpha_pool):
    """
    Update the composite alpha based on the alpha pool.
    
    For each alpha expression in alpha_pool (each a list of tokens), evaluate it 
    on the market data to obtain a signal series. Then form a design matrix X 
    (each column is one alpha's signal) and target vector y as the next-day returns 
    from "^FTSE_Close". Solve a linear regression problem (OLS) to determine 
    the weights that minimize MSE between X*w and y. The composite alpha signal is 
    computed as X*w and its IC (Pearson correlation with y) is returned.
    
    If the pool size exceeds a threshold K (e.g. 10), the alpha with the smallest 
    absolute weight is pruned and the regression is re-run.
    """
    
    # Load market data
    df = pd.read_csv("ftse100_data_wide.csv", index_col=0, parse_dates=True, low_memory=False)
    
    signals = []
    valid_expressions = []
    # Evaluate each alpha in the pool.
    for expr in alpha_pool:
        try:
            signal = evaluate_rpn(expr, df)
            signals.append(signal)
            valid_expressions.append(expr)
        except Exception as e:
            print(f"Error evaluating alpha expression {expr}: {e}")
            continue
    # Update the alpha_pool to only include those that evaluated successfully.
    alpha_pool[:] = valid_expressions
    if len(signals) == 0:
        return 0.0
    
    # Create a DataFrame where each column is one alpha's signal.
    signals_df = pd.concat(signals, axis=1, join="inner").dropna()
    if signals_df.empty:
        return 0.0
    # Rename columns for clarity.
    signals_df.columns = [f"a{i}" for i in range(signals_df.shape[1])]
    
    # Compute future returns r_{t+1} using "^FTSE_Close"
    if "^FTSE_Close" not in df.columns:
        raise ValueError("Column '^FTSE_Close' not found in data.")
    close = df["^FTSE_Close"]
    r = close.pct_change(fill_method=None).shift(-1)
    r = r.loc[signals_df.index]
    
    # Combine signals and returns, dropping any missing values.
    combined = pd.concat([signals_df, r], axis=1, join="inner").dropna()
    if combined.empty:
        return 0.0
    X = combined.iloc[:, :-1].values  # alpha signals
    y = combined.iloc[:, -1].values   # future returns
    
    # Solve for weights using ordinary least squares.
    try:
        w, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    except Exception as e:
        print("Error in linear regression for composite alpha:", e)
        return 0.0
    
    # Compute the composite alpha signal.
    composite = np.dot(X, w)
    # Compute the composite IC as the Pearson correlation between composite and y.
    std_composite = np.std(composite, ddof=1)
    std_y = np.std(y, ddof=1)
    if std_composite == 0 or std_y == 0:
        ic = 0.0
    else:
        ic = np.corrcoef(composite, y)[0, 1]
    
    # If the pool is too large, prune the alpha with the smallest absolute weight.
    K = 10  # threshold
    if len(w) > K:
        idx_to_prune = np.argmin(np.abs(w))
        del alpha_pool[idx_to_prune]
        # Re-run update_composite_alpha after pruning.
        return update_composite_alpha(alpha_pool)
    
    return ic

###############################################################################
# 4. AlphaGame Class for Constructing Alpha Formulas (RPN)
###############################################################################
class AlphaGame:
    """
    Environment for building an alpha formula in RPN.
    - Begins with [BEG] and ends when [END] is appended.
    - Provides intermediate rewards (IC minus redundancy penalty) when valid.
    - Terminal reward (composite alpha IC) is computed when the expression is complete.
    """
    def __init__(self, alpha_pool=None, lam=0.1, max_length=10):
        self.alpha_pool = alpha_pool if alpha_pool is not None else []
        self.lam = lam
        self.max_length = max_length
        self.rpn_expression = [BEG]
        self._terminal = False
        self._cached_reward = None

    def get_possible_moves(self):
        if self._terminal:
            return []
        current_length = len(self.rpn_expression)
        if current_length >= self.max_length:
            if self._is_valid_partial(self.rpn_expression):
                return [END]
            else:
                return []
        possible_moves = []
        stack_size = self._rpn_stack_size(self.rpn_expression)
        for opnd in OPERANDS:
            if current_length < self.max_length:
                possible_moves.append(opnd)
        for op, arity in OPERATORS.items():
            if stack_size >= arity and current_length < self.max_length:
                possible_moves.append(op)
        if stack_size == 1 and current_length > 1:
            possible_moves.append(END)
        return possible_moves

    def make_move(self, move):
        if self._terminal:
            return
        self.rpn_expression.append(move)
        if move == END:
            self._terminal = True
        self._cached_reward = None

    def is_terminal(self):
        return self._terminal

    def get_reward(self):
        if self._cached_reward is not None:
            return self._cached_reward
        if self._terminal:
            final_alpha = self._trim_expression(self.rpn_expression)
            self.alpha_pool.append(final_alpha)
            final_ic = update_composite_alpha(self.alpha_pool)
            self._cached_reward = final_ic
            return final_ic
        if self._is_valid_partial(self.rpn_expression):
            partial_alpha = self._trim_expression(self.rpn_expression)
            k = len(self.alpha_pool)
            alpha_ic = compute_ic(partial_alpha)
            redundancy_penalty = 0.0
            if k > 0:
                mutic_sum = sum(compute_mutic(partial_alpha, alpha_i) for alpha_i in self.alpha_pool)
                redundancy_penalty = self.lam * (mutic_sum / k)
            self._cached_reward = alpha_ic - redundancy_penalty
            return self._cached_reward
        else:
            self._cached_reward = 0.0
            return 0.0

    def _trim_expression(self, tokens):
        return [t for t in tokens if t not in (BEG, END)]

    def _is_valid_partial(self, rpn_tokens):
        core_tokens = [t for t in rpn_tokens if t != BEG]
        if core_tokens and core_tokens[-1] == END:
            core_tokens = core_tokens[:-1]
        return self._rpn_stack_size(core_tokens) == 1

    def _rpn_stack_size(self, tokens):
        stack_count = 0
        for token in tokens:
            if token in (BEG, END):
                continue
            elif token in OPERATORS:
                arity = OPERATORS[token]
                stack_count -= arity
                if stack_count < 0:
                    return -1
                stack_count += 1
            else:
                stack_count += 1
        return stack_count

    def print_expression(self):
        print("RPN expression:", self.rpn_expression)
        print()

###############################################################################
# 5. MCTSNode Class for PUCT Search
###############################################################################
class MCTSNode:
    def __init__(self, state, parent=None, move=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.move = move
        self.prior = prior
        self.wins = 0.0
        self.visits = 0
        self.children = []
        self.untried_moves = state.get_possible_moves()
        self.expanded = False

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def add_child(self, move, state, prior):
        child_node = MCTSNode(state, parent=self, move=move, prior=prior)
        self.children.append(child_node)
        if move in self.untried_moves:
            self.untried_moves.remove(move)
        return child_node

    def best_child(self, c_param=1.0):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            Q = child.wins / child.visits if child.visits > 0 else 0.0
            U = c_param * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            score = Q + U
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def update(self, reward):
        self.visits += 1
        self.wins += reward

###############################################################################
# 6. MCTS with PUCT Function
###############################################################################
def evaluate_state(state, network, vocab, device='cpu'):
    test = True
    if test == True:
        legal_moves = state.get_possible_moves()
        if legal_moves:
            policy = {move: 1.0 / len(legal_moves) for move in legal_moves}
        else:
            policy = {}
        value = state.get_reward()
        return policy, value
    else:
        # Convert state to tensor representation
        state_tensor = state_to_tensor(state, vocab).to(device)
        # Get network outputs
        pred_policy, pred_value = network(state_tensor)
        # Convert the predicted policy (a probability distribution) into a dict mapping legal moves to probabilities
        legal_moves = state.get_possible_moves()
        policy = {}
        if legal_moves:
            # Suppose you have a mapping from token indices to move tokens.
            # For each legal move, get its corresponding probability from the network's output.
            for move in legal_moves:
                move_str = str(move)  # ensure it's a string key
                if move_str in vocab:
                    idx = vocab[move_str]
                    policy[move] = pred_policy[0, idx].item()  # assuming batch size 1
        value = pred_value.item()
        return policy, value

def mcts_puct(root_state, network, itermax, vocab, c_param=1.4, verbose=False):
    root_node = MCTSNode(root_state, parent=None, move=None, prior=0.0)
    for i in range(itermax):
        node = root_node
        state = copy.deepcopy(root_state)
        path = [node]

        while True:
            if state.is_terminal():
                print(i, state.rpn_expression)
                reward = state.get_reward()
                break
            if not node.expanded:
                policy, value = evaluate_state(state, network, vocab)
                legal_moves = state.get_possible_moves()
                for move in legal_moves:
                    next_state = copy.deepcopy(state)
                    next_state.make_move(move)
                    node.add_child(move, next_state, prior=policy.get(move, 0))
                node.expanded = True
                reward = value
                print(i, next_state.rpn_expression)
                break
            else:
                node = node.best_child(c_param)
                path.append(node)
                state.make_move(node.move)

        for n in path:
            n.update(reward)

        if verbose:
            ucb_values = {}
            for child in root_node.children:
                Q = child.wins / child.visits if child.visits > 0 else 0.0
                U = c_param * child.prior * math.sqrt(root_node.visits) / (1 + child.visits)
                ucb_values[str(child.move)] = Q + U
            print(f"Iteration {i}: UCB values at root: {ucb_values}")
    return root_node

###############################################################################
# 7. Neural Architecture: Policy & Value Network (AlphaMiningNet)
###############################################################################
class AlphaMiningNet(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 32, hidden_dim: int = 64,
                 num_layers: int = 4, mlp_hidden_dim: int = 32, action_dim: int = None):
        super().__init__()
        if action_dim is None:
            action_dim = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, action_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)
        )

    def forward(self, token_ids: torch.Tensor):
        embedded = self.embedding(token_ids)
        outputs, hidden = self.gru(embedded)
        final_hidden = hidden[-1]
        logits = self.policy_head(final_hidden)
        policy = F.softmax(logits, dim=-1)
        value = self.value_head(final_hidden)
        return policy, value

    def get_action_probs(self, token_ids: torch.Tensor) -> torch.Tensor:
        policy, _ = self.forward(token_ids)
        return policy

###############################################################################
# 8. Utility: Convert State to Tensor using a Vocabulary Mapping
###############################################################################
def state_to_tensor(state, vocab):
    tokens = [str(token) for token in state.rpn_expression]
    indices = [vocab[token] for token in tokens if token in vocab]
    return torch.tensor([indices], dtype=torch.long)

###############################################################################
# 9. Self-Play Game to Generate Training Examples
###############################################################################
def self_play_game(network, mcts_iterations, c_param, vocab):
    state = AlphaGame()
    examples = []
    while not state.is_terminal():
        state_tensor = state_to_tensor(state, vocab)
        root_node = mcts_puct(state, network, mcts_iterations, vocab, c_param, verbose=False)
        total_visits = sum(child.visits for child in root_node.children) + 1e-8
        target_policy = torch.zeros(len(vocab), dtype=torch.float32)
        for child in root_node.children:
            token_str = str(child.move)
            if token_str in vocab:
                target_policy[vocab[token_str]] = child.visits / total_visits
        target_value = state.get_reward()
        examples.append((state_tensor, target_policy, target_value))
        best_child = max(root_node.children, key=lambda c: c.visits)
        state.make_move(best_child.move)
    return examples

###############################################################################
# 10. Training Function for the Network
###############################################################################
from torch.nn.utils.rnn import pad_sequence

def train_network(network, optimizer, training_examples, epochs=1, batch_size=16, device='cpu', pad_idx=0):
    """
    Train the network using self-play examples.
    
    Each training example is a tuple (state_tensor, target_policy, target_value),
    where state_tensor is a 2D tensor of shape [1, seq_len]. Since seq_len may vary
    across examples, we pad them to a common length using pad_sequence.
    
    Args:
        network: The policy/value network.
        optimizer: The optimizer.
        training_examples: A list of training examples.
        epochs: Number of epochs.
        batch_size: Batch size.
        device: Torch device.
        pad_idx: The token index used for padding.
    
    Returns:
        total_loss: Cumulative loss over all batches.
    """
    network.train()
    total_loss = 0.0
    random.shuffle(training_examples)
    
    for epoch in range(epochs):
        for i in range(0, len(training_examples), batch_size):
            batch = training_examples[i:i+batch_size]
            
            # Each example[0] is a tensor of shape [1, seq_len]. Remove the extra dim.
            state_tensors = [ex[0].squeeze(0) for ex in batch]
            # Pad the sequences to the same length.
            padded_state_batch = pad_sequence(state_tensors, batch_first=True, padding_value=pad_idx)
            padded_state_batch = padded_state_batch.to(device)
            
            target_policy_batch = torch.stack([example[1] for example in batch]).to(device)
            target_value_batch = torch.tensor([example[2] for example in batch], dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            pred_policy, pred_value = network(padded_state_batch)
            eps = 1e-8
            loss_policy = -torch.mean(torch.sum(target_policy_batch * torch.log(pred_policy + eps), dim=1))
            loss_value = torch.mean((target_value_batch - pred_value.squeeze()) ** 2)
            loss = loss_policy + loss_value
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return total_loss


###############################################################################
# 11. Inference: Construct the Final Alpha and Update the Composite Alpha
###############################################################################
def inference_alpha(network, mcts_iterations, c_param, vocab):
    state = AlphaGame()
    move_num = 1
    while not state.is_terminal():
        print(f"Step {move_num}: Current expression: {state.rpn_expression}")
        root_node = mcts_puct(state, network, mcts_iterations, vocab, c_param, verbose=False)
        best_child = max(root_node.children, key=lambda c: c.visits)
        chosen_move = best_child.move
        print(f"Selected token: {chosen_move}")
        state.make_move(chosen_move)
        move_num += 1
    print("\nFinal alpha expression:", state.rpn_expression)
    final_reward = state.get_reward()
    print(f"Composite alpha IC (reward): {final_reward:.4f}")
    print("Alpha pool:", state.alpha_pool)
    return state

###############################################################################
# 12. Main: Self-Play Training and Alpha Composite Inference
###############################################################################
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build vocabulary: include string versions of OPERANDS, OPERATORS, BEG, END.
    operator_keys = list(OPERATORS.keys())
    vocab_tokens = [str(token) for token in OPERANDS] + operator_keys + [BEG, END]
    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
    vocab_size = len(vocab)
    
    network = AlphaMiningNet(vocab_size=vocab_size, action_dim=vocab_size).to(device)
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    
    mcts_iterations = 500
    c_param = 1.4
    num_self_play_games = 50
    training_epochs = 5

    print("Starting self-play training...\n")
    for game_num in range(num_self_play_games):
        print(f"Self-play game {game_num+1}/{num_self_play_games}...")
        training_examples = self_play_game(network, mcts_iterations, c_param, vocab)
        loss = train_network(network, optimizer, training_examples, epochs=training_epochs, device=device)
        print(f"Game {game_num+1} training completed. Loss: {loss:.4f}\n")
    
    print("Starting inference: Computing the composite alpha...\n")
    final_state = inference_alpha(network, mcts_iterations, c_param, vocab)
