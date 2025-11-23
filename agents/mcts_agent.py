from agents.agent import Agent
from store import register_agent
import sys
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

import math
import random
import numpy as np

class MCTSNode:
    def __init__(self, board, parent=None, move=None, player_just_moved=None):
        self.board = board
        self.parent = parent
        self.move = move  # The move that led to this board state
        self.player_just_moved = player_just_moved
        self.children = []
        self.wins = 0
        self.visits = 0

        self.next_player = 3 - player_just_moved if player_just_moved else None

        self.untried_moves = get_valid_moves(board, self.next_player)

    def uct_select_child(self):
        """
        Selects a child node using the UCB1 formula.
        """
        c = 1.0

        s = sorted(self.children, key=lambda child:
            (child.wins / child.visits) +
            c * math.sqrt(math.log(self.visits) / child.visits)
        )

        return s[-1]

    def add_child(self, move, board_state):
        """
        Removes the move from untried_moves and creates a new child node.
        """
        node = MCTSNode(
            board=board_state,
            parent=self,
            move=move,
            player_just_moved=self.next_player
        )
        self.untried_moves.remove(move)
        self.children.append(node)
        return node

    def update(self, result):
        """
        Update the node statistics.
        result: 1 if the player_just_moved WON, 0 if they LOST.
        """
        self.visits += 1
        self.wins += result

@register_agent("mcts_agent")
class MctsAgent(Agent):
    def __init__(self):
        super().__init__()
        self.name = "MCTS_Agent"
        self.time_limit = 1.9  # strict cutoff to ensure we don't crash the 2s limit

    def step(self, board, color, opponent):
        start_time = time.time()

        root = MCTSNode(board, parent=None, move=None, player_just_moved=opponent)

        if not root.untried_moves:
            return None

        iterations = 0
        while time.time() - start_time < self.time_limit:
            iterations += 1
            node = root

            while node.untried_moves == [] and node.children != []:
                node = node.uct_select_child()

            if node.untried_moves != []:
                move = random.choice(node.untried_moves)

                # Create the new board state
                temp_board = node.board.copy()
                execute_move(temp_board, move, node.next_player)

                node = node.add_child(move, temp_board)

            # Play a random game from this state to the end
            winner = self.simulate_random_game(node.board, node.next_player)

            #BACKPROPAGATION
            while node is not None:
                if winner == node.player_just_moved:
                    node.update(1)
                else:
                    node.update(0)

                node = node.parent

        if not root.children:
            return random.choice(get_valid_moves(board, color))

        best_node = sorted(root.children, key=lambda c: c.visits)[-1]

        return best_node.move

    def simulate_random_game(self, board, current_player):
        """
        Simulates a game by playing random moves until the end or a cutoff.
        """
        sim_board = board.copy()
        player = current_player
        moves_count = 0
        max_moves = 10

        while moves_count < max_moves:
            moves = get_valid_moves(sim_board, player)

            if not moves:
                # If current player has no moves, check opponent
                opp_moves = get_valid_moves(sim_board, 3 - player)
                if not opp_moves:
                    break
                player = 3 - player
                continue

            # Pick a random move
            move = random.choice(moves)
            execute_move(sim_board, move, player)

            player = 3 - player
            moves_count += 1

        p1 = np.sum(sim_board == 1)
        p2 = np.sum(sim_board == 2)

        if p1 > p2: return 1
        if p2 > p1: return 2
        return -1