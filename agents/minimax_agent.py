from agents.agent import Agent
from store import register_agent
import sys
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

import time
import numpy as np
import random

@register_agent("minimax_agent")
class MinimaxAgent(Agent):
    def __init__(self):
        super().__init__()
        self.name = "Winner_Agent"
        self.time_limit = 1.9
        self.start_time = 0

    def step(self, board, color, opponent):
        self.start_time = time.time()
        legal_moves = get_valid_moves(board, color)

        if not legal_moves:
            return None

        best_move = random.choice(legal_moves)  # Fallback

        # ITERATIVE DEEPENING
        try:
            for depth in range(2, 10):

                # Check time before starting a new depth
                if time.time() - self.start_time > self.time_limit:
                    break

                score, move = self.minimax(board, depth, float('-inf'), float('inf'), True, color, opponent)

                if time.time() - self.start_time < self.time_limit:
                    best_move = move
                    print(f"Depth {depth} completed. Score: {score}")
                else:
                    break  # Timeout

        except TimeoutError:
            pass

        return best_move

    def minimax(self, board, depth, alpha, beta, maximizing_player, color, opponent):
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError

        valid_moves = get_valid_moves(board, color if maximizing_player else opponent)

        if depth == 0 or not valid_moves:
            return self.evaluate_board(board, color, opponent), None

        best_move = None

        if maximizing_player:
            max_eval = float('-inf')
            for move in valid_moves:
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

    def evaluate_board(self, board, color, opponent):
        my_discs = np.sum(board == color)
        opp_discs = np.sum(board == opponent)
        empty = np.sum(board == 0)

        if empty > 15: # Early Game: Focus on spreading
            score = (my_discs * 1.0) - (opp_discs * 1.2)

        else: # Late Game: Focus purely on piece count
            score = my_discs - opp_discs

        # Corner bias
#        size = board.shape[0]
#        corners = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
#
#        corner_val = 5.0
#        for r, c in corners:
#            piece = board[r, c]
#            if piece == color:
#                score += corner_val
#            elif piece == opponent:
#                score -= corner_val

        return score