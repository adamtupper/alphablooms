"""A board class for the game of Blooms.
"""
from itertools import permutations

import numpy as np


class Board:
    """A board class for the game of Blooms.
    """
    def __init__(self, size=4, score_target=15):
        """Initialise a new game board.

        The state of the board is represented by a 3D Numpy array, where the
        first dimension has four elements (one for each color stone) and the
        second an third dimensions represent a position on the board (in axial
        coordinates).

        :param size: the size of the board (either base 4, 5, or 6).
        :param score_target: the number of 'captures' to win the game. It is
            recommended that the number of captures are 15 for a base 4 board,
            20 for a base 5 board, and 25 or 30 for a base 6 board.
        """
        self.size = size
        self.score_target = score_target
        self.captures = [0, 0]
        self.board = np.zeros((4, 2 * self.size - 1, 2 * self.size - 1))

    def get_legal_moves(self, player):
        """Returns all the legal moves for the given player.

        Each turn, the player can place up to two stones on any empty spaces on
        the board. However, if the player places two stones they must be
        different colors.

        :param player: 0 or 1 to denote the player in question.
        :return: the list of all legal moves for the given player.
        """
        # Convert the board to a 2D representation where each index (q, r) is
        # 1 if there is a piece on that space or 0 otherwise.
        flattened_board = np.sum(self.board, axis=0)

        # Find the list of empty spaces
        empty_spaces = []
        for r in range(flattened_board.shape[1]):
            for q in range(flattened_board.shape[0]):
                if (q + r >= self.size - 1) and (4 * self.size - 4 - q - r >= self.size - 1):
                    # (q, r) is a valid space
                    if flattened_board[r][q] == 0:
                        # The space is empty
                        empty_spaces.append((q, r))

        # Generate all possible one stone moves
        moves = []
        moves += [[(q, r, 0), ()] for (q, r) in empty_spaces]  # All possible one stone moves of Colour 0
        moves += [[(q, r, 1), ()] for (q, r) in empty_spaces]  # All possible one stone moves of Colour 1

        # Generate all possible two stone moves
        moves += [[(a[0], a[1], 0), (b[0], b[1], 1)] for (a, b) in permutations(empty_spaces, r=2)]

        return moves

    def has_legal_moves(self):
        """Return True or False depending on whether there are any legal moves
        remaining.

        There are legal moves remaining if there are empty spaces on the board.

        :return: True if there are legal moves, False otherwise.
        """
        # Convert the board to a 2D representation where each index (q, r) is
        # 1 if there is a piece on that space or 0 otherwise.
        flattened_board = np.sum(self.board, axis=0)

        # Check each index (q, r) to see if it is empty.
        for r in range(flattened_board.shape[1]):
            for q in range(flattened_board.shape[0]):
                if (q + r >= self.size - 1) and (4 * self.size - 4 - q - r >= self.size - 1):
                    # (q, r) is a valid space
                    if flattened_board[r][q] == 0:
                        # The space is empty
                        return True

    def is_win(self, player):
        """A player wins if they reach the target number of captures.

        :param player: 0 or 1 to denote the player in question.
        :return: True if the given player has won the game, False otherwise.
        """
        return self.captures[player] >= self.score_target

    def execute_move(self, move, player):
        """Perform the given move on the board.

        :param move: the move to be performed.
        :param player: the player performing the move (0 or 1).
        """
        raise NotImplementedError
