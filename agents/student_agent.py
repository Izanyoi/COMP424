# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"
    self.max_depth = 3

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """
    # 1. Get valid moves
    legal_moves = get_valid_moves(chess_board, player)

    if not legal_moves:
      return None

    # 2. Run Minimax with Alpha-Beta Pruning
    # alpha = -infinity, beta = +infinity
    _, best_move = self.minimax(chess_board, self.max_depth, float('-inf'), float('inf'), True, player, opponent)

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return best_move

  def minimax(self, board, depth, alpha, beta, maximizing_player, color, opponent):
    """
    Standard Minimax with Alpha-Beta pruning.
    """
    # Base case: Depth 0
    if depth == 0:
      return self.evaluate_board_fast(board, color, opponent), None

    current_player = color if maximizing_player else opponent
    valid_moves = get_valid_moves(board, current_player)

    # Base case: No moves (Game End or Pass Turn)
    if not valid_moves:
      return self.evaluate_board_fast(board, color, opponent), None

    best_move = None

    if maximizing_player:
      max_eval = float('-inf')
      for move in valid_moves:
        # 1. Fast Copy (Numpy specific)
        sim_board = board.copy()

        # 2. Execute Move
        execute_move(sim_board, move, color)

        # 3. Recurse
        eval_score, _ = self.minimax(sim_board, depth - 1, alpha, beta, False, color, opponent)

        if eval_score > max_eval:
          max_eval = eval_score
          best_move = move

        # 4. Prune
        alpha = max(alpha, eval_score)
        if beta <= alpha:
          break
      return max_eval, best_move

    else:
      min_eval = float('inf')
      for move in valid_moves:
        sim_board = board.copy()
        execute_move(sim_board, move, opponent)

        eval_score, _ = self.minimax(sim_board, depth - 1, alpha, beta, True, color, opponent)

        if eval_score < min_eval:
          min_eval = eval_score
          best_move = move

        beta = min(beta, eval_score)
        if beta <= alpha:
          break
      return min_eval, best_move

  def evaluate_board_fast(self, board, color, opponent):
    """
    A purely vectorized evaluation function.
    NO loops, NO move generation. purely numpy math.
    """
    # --- 1. Material Count ---
    # This is extremely fast in Numpy
    my_discs = np.sum(board == color)
    opp_discs = np.sum(board == opponent)

    # --- 2. Corner Control ---
    # Check corners manually
    n = board.shape[0] - 1
    corners = [(0, 0), (0, n), (n, 0), (n, n)]
    corner_score = 0
    for r, c in corners:
      if board[r, c] == color:
        corner_score += 5
      elif board[r, c] == opponent:
        corner_score -= 5

    # --- 3. Strategy Weights ---
    # Simple Logic: If I am winning significantly, play safe.
    # If I am losing, play aggressive.
    material_score = my_discs - opp_discs

    return material_score + corner_score