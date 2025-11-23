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
    super().__init__()
    self.name = "Locality_Pruning_Agent"
    self.time_limit = 1.8
    self.start_time = 0

    # Pre-calculate offsets for a 5x5 "danger zone" (range 2)
    self.offsets = []
    for r in range(-3, 4):
        for c in range(-3, 4):
            self.offsets.append((r, c))

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

    def __init__(self):
        super().__init__()
        self.name = "Student_Agent"
        self.time_limit = 1.9
        self.start_time = 0
        # Pre-calculate offsets for a 5x5 "danger zone" (range 2)
        self.offsets = []
        for r in range(-2, 3):
            for c in range(-2, 3):
                self.offsets.append((r, c))

  def step(self, board, color, opponent):
      self.start_time = time.time()

      legal_moves = get_valid_moves(board, color)

      if not legal_moves:
          return None

      if len(legal_moves) == 1:
          return legal_moves[0]

      best_move = legal_moves[0]

      try:
          # Iterative Deepening
          for depth in range(1, 20):
              if time.time() - self.start_time > self.time_limit:
                  break

              score, move = self.minimax(board, depth, float('-inf'), float('inf'), True, color, opponent)

              if time.time() - self.start_time < self.time_limit:
                  if move is not None:
                      best_move = move
              else:
                  break
      except TimeoutError:
          pass  # Timeout triggered, return best result so far

      return best_move

  def minimax(self, board, depth, alpha, beta, maximizing_player, color, opponent):
      if time.time() - self.start_time > self.time_limit:
          raise TimeoutError

      current_player = color if maximizing_player else opponent
      valid_moves = get_valid_moves(board, current_player)

      if depth == 0 or not valid_moves:
          return self.evaluate_board(board, color, opponent), None

      moves_to_search = []

      # Only prune if branching factor is high and not at root
      if len(valid_moves) > 8 and depth > 1:
          target_pieces_color = opponent if maximizing_player else color
          targets = np.argwhere(board == target_pieces_color)

          hot_zones = set()
          rows, cols = board.shape

          for tr, tc in targets:
              for dr, dc in self.offsets:
                  nr, nc = tr + dr, tc + dc
                  if 0 <= nr < rows and 0 <= nc < cols:
                      hot_zones.add((nr, nc))

          urgent_moves = []
          quiet_moves = []

          for move in valid_moves:
              # Safe destination extraction
              dest = self.get_move_destination(move)

              if dest in hot_zones:
                  urgent_moves.append(move)
              else:
                  quiet_moves.append(move)

          moves_to_search.extend(urgent_moves)

          if quiet_moves:
              # Safe sort using the robust is_jump function
              # We try to put clones (not jumps) first among quiet moves
              quiet_moves.sort(key=lambda m: self.is_jump(m))
              moves_to_search.extend(quiet_moves[:2])

          if not moves_to_search:
              moves_to_search = valid_moves
      else:
          moves_to_search = valid_moves

      # --- Standard Minimax ---
      best_move = None

      if maximizing_player:
          max_eval = float('-inf')
          for move in moves_to_search:
              sim_board = board.copy()
              execute_move(sim_board, move, color)
              eval_score, _ = self.minimax(sim_board, depth - 1, alpha, beta, False, color, opponent)

              if eval_score > max_eval:
                  max_eval = eval_score
                  best_move = move
              alpha = max(alpha, eval_score)
              if beta <= alpha: break
          return max_eval, best_move
      else:
          min_eval = float('inf')
          for move in moves_to_search:
              sim_board = board.copy()
              execute_move(sim_board, move, opponent)
              eval_score, _ = self.minimax(sim_board, depth - 1, alpha, beta, True, color, opponent)

              if eval_score < min_eval:
                  min_eval = eval_score
                  best_move = move
              beta = min(beta, eval_score)
              if beta <= alpha: break
          return min_eval, best_move

  def evaluate_board(self, board, color, opponent):
      my_discs = np.sum(board == color)
      opp_discs = np.sum(board == opponent)
      # Simple material score is robust and fast
      return my_discs - opp_discs

  # --- ROBUST HELPER FUNCTIONS ---

  def get_move_destination(self, move):
      """
    Safely extracts the destination (r, c) from any move object/tuple.
    """
      try:
          # 1. Try common object attributes
          if hasattr(move, 'end_row') and hasattr(move, 'end_col'):
              return (move.end_row, move.end_col)
          if hasattr(move, 'dest'):  # Often a tuple (r,c)
              return move.dest
          if hasattr(move, 'end'):
              return move.end

          # 2. Try Indexing (Tuples/Lists)
          # Format: ((r1, c1), (r2, c2))
          if len(move) == 2 and isinstance(move[0], (tuple, list)):
              return move[1]
          # Format: (r1, c1, r2, c2)
          if len(move) == 4 and isinstance(move[0], int):
              return (move[2], move[3])

      except Exception:
          pass

      # Fallback: If we can't parse, return dummy to treat as non-hot
      return (-1, -1)

  def is_jump(self, move):
      """
    Determines if a move is a jump (dist > 1) safely.
    Returns True if Jump, False if Clone.
    """
      try:
          r1, c1, r2, c2 = 0, 0, 0, 0

          # 1. Try common object attributes
          if hasattr(move, 'start_row'):
              r1, c1 = move.start_row, move.start_col
              r2, c2 = move.end_row, move.end_col
          elif hasattr(move, 'src') and hasattr(move, 'dest'):
              r1, c1 = move.src
              r2, c2 = move.dest

          # 2. Try Indexing
          elif isinstance(move, (tuple, list)):
              if len(move) == 2:  # ((r1,c1), (r2,c2))
                  r1, c1 = move[0]
                  r2, c2 = move[1]
              elif len(move) == 4:  # (r1, c1, r2, c2)
                  r1, c1, r2, c2 = move

          # Calculate distance
          dist = max(abs(r1 - r2), abs(c1 - c2))
          return dist > 1

      except Exception:
          return False