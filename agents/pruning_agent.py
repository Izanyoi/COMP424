from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

import random

@register_agent("advanced_agent")
class AdvancedAgent(Agent):
    """
    Advanced Ataxx Agent using Iterative Deepening, Move Ordering, and Beam Search.
    Fixed for MoveCoordinates object attribute access.
    """

    def __init__(self):
        super().__init__()
        self.name = "Beam_Search_Agent"
        self.time_limit = 1.85
        self.start_time = 0
        self.beam_width = 10

    def step(self, board, color, opponent):
        self.start_time = time.time()

        legal_moves = get_valid_moves(board, color)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        best_move = random.choice(legal_moves)

        try:
            # Iterative Deepening
            for depth in range(1, 20):

                # Sort moves using the FIXED order_moves function
                if depth == 1:
                    legal_moves = self.order_moves(board, legal_moves, color, opponent)

                score, move = self.minimax(board, depth, float('-inf'), float('inf'), True, color, opponent)

                if time.time() - self.start_time < self.time_limit:
                    best_move = move
                else:
                    break
        except TimeoutError:
            pass

        return best_move

    def order_moves(self, board, moves, color, opponent):
        """
        Heuristic to sort moves.
        Fixed to handle MoveCoordinates objects.
        """
        scores = []
        for move in moves:
            # --- FIXED COORDINATE EXTRACTION ---
            # We try the most common attribute names for this course/game structure.
            try:
                r1 = move.src_row
                c1 = move.src_col
                r2 = move.dest_row
                c2 = move.dest_col
            except AttributeError:
                # Fallback if attributes are named differently (e.g., start_row, r1, etc.)
                # You can inspect the object with `dir(move)` if this still fails.
                # For now, we try a generic fallback for safety:
                try:
                    r1, c1, r2, c2 = move.r1, move.c1, move.r2, move.c2
                except AttributeError:
                    # Final fallback: assumes attributes might be just row/col properties
                    # If this hits, the class structure is unique.
                    r1, c1, r2, c2 = 0, 0, 0, 0

                    # 1. Check Capture Potential
            capture_score = 0

            # Bounds check for the destination neighborhood
            r_min, r_max = max(0, r2 - 1), min(board.shape[0] - 1, r2 + 1)
            c_min, c_max = max(0, c2 - 1), min(board.shape[1] - 1, c2 + 1)

            # Slice the 3x3 area around destination
            region = board[r_min:r_max + 1, c_min:c_max + 1]

            # Count opponents in this region (this is the capture amount)
            capture_score = np.count_nonzero(region == opponent)

            # 2. Check Clone vs Jump
            dist = max(abs(r1 - r2), abs(c1 - c2))
            is_clone = (dist == 1)

            # 3. Combined Score
            # Captures are high value (10 pts). Clones are medium value (5 pts).
            move_score = (capture_score * 10) + (5 if is_clone else 0)

            scores.append(move_score)

        # Sort moves based on the scores we calculated
        sorted_moves = [m for _, m in sorted(zip(scores, moves), key=lambda pair: pair[0], reverse=True)]
        return sorted_moves

    def minimax(self, board, depth, alpha, beta, maximizing_player, color, opponent):
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError

        if depth == 0:
            return self.evaluate_board(board, color, opponent), None

        current_player = color if maximizing_player else opponent
        valid_moves = get_valid_moves(board, current_player)

        if not valid_moves:
            return self.evaluate_board(board, color, opponent), None

        # Order moves for efficiency
        ordered_moves = self.order_moves(board, valid_moves, current_player, color if maximizing_player else opponent)

        # Beam Search Slice
        if len(ordered_moves) > self.beam_width:
            ordered_moves = ordered_moves[:self.beam_width]

        best_move = None

        if maximizing_player:
            max_eval = float('-inf')
            for move in ordered_moves:
                sim_board = board.copy()
                execute_move(sim_board, move, color)

                eval_score, _ = self.minimax(sim_board, depth - 1, alpha, beta, False, color, opponent)

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move

        else:
            min_eval = float('inf')
            for move in ordered_moves:
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

    def evaluate_board(self, board, color, opponent):
        # Fast Vectorized Evaluation
        my_discs = np.sum(board == color)
        opp_discs = np.sum(board == opponent)
        return my_discs - opp_discs