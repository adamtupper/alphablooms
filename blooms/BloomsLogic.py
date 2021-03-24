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
        self.board_2d = np.zeros((2 * self.size - 1, 2 * self.size - 1))
        self.colours = [(1, 2), (3, 4)]

    def get_board_3d(self):
        """Converts the board representation into a 3D representation, where
        each channel stores the pieces of a different colour. This
        representation is used as input to a CNN.

        :return: A 3D representation of the board state, which is a 3D Numpy
            array with shape (4, 2n - 1, 2n - 1).
        """
        board_3d = np.repeat(self.board_2d[np.newaxis, :, :], 4, axis=0)
        for c in range(0, board_3d.shape[0]):
            board_3d[c] = np.where(board_3d[c] == c + 1, 1, 0)

        return board_3d

    def get_empty_spaces(self):
        """Returns a list of all the empty spaces on the board.

        :return: the list of empty spaces on the board. Each element is the
            coordinate of the space, i.e. (q, r).
        """
        empty_spaces = []
        for r in range(self.board_2d.shape[0]):
            for q in range(self.board_2d.shape[1]):
                if self.is_valid_space((q, r)) and self.board_2d[r][q] == 0:
                    # (q, r) is a valid space and is empty
                    empty_spaces.append((q, r))

        return empty_spaces

    def is_valid_space(self, position):
        """Check to see if the given position is a valid space on the board.

        Because of the hexagonal shape of the board, some elements of the 2D
        board representation are not spaces on the board.

        :param position:  A tuple representing the (q, r) coord to place the
            stone.
        :return: True if the given position is a valid space, False otherwise.
        """
        q, r = position
        not_in_top_left = q + r >= self.size - 1
        not_in_bottom_right = 4 * self.size - 4 - q - r >= self.size - 1

        return not_in_top_left and not_in_bottom_right

    def place_stone(self, position, colour):
        """Place a stone on the board.

        :param position: A tuple representing the (q, r) coord to place the
            stone.
        :param colour: The colour of the stone to be placed (1, 2, 3, or 4).
        """
        q, r = position

        # Check the position is valid and empty
        assert self.is_valid_space(position)
        assert self.board_2d[r, q] == 0

        self.board_2d[r, q] = colour

    def get_legal_moves(self, player):
        """Returns all the legal moves for the given player.

        Each turn, the player can place up to two stones on any empty spaces on
        the board. However, if the player places two stones they must be
        different colors.

        :param player: 0 or 1 to denote the player in question.
        :return: the list of all legal moves for the given player.
        """
        # Get the channels where the positions of the player's different colour stones are stored
        colour1, colour2 = self.colours[player]
        c1, c2 = colour1 - 1, colour2 - 1

        # Convert the board to a 2D representation where each index (q, r) is
        # 1 if there is a piece on that space or 0 otherwise.
        flattened_board = np.sum(self.board, axis=0)

        # Find the list of empty spaces
        empty_spaces = []
        for r in range(flattened_board.shape[2]):
            for q in range(flattened_board.shape[1]):
                if self.is_valid_space((q, r)):
                    # (q, r) is a valid space
                    if flattened_board[r][q] == 0:
                        # The space is empty
                        empty_spaces.append((q, r))

        # Generate all possible one stone moves
        moves = []
        moves += [[(q, r, c1), ()] for (q, r) in empty_spaces]  # All possible one stone moves of Colour 0
        moves += [[(q, r, c2), ()] for (q, r) in empty_spaces]  # All possible one stone moves of Colour 1

        # Generate all possible two stone moves
        moves += [[(m1[0], m1[1], c1), (m2[0], m2[1], c2)] for (m1, m2) in permutations(empty_spaces, r=2)]

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
        for r in range(self.board_2d.shape[0]):
            for q in range(self.board_2d.shape[1]):
                if self.is_valid_space((q, r)):
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

        :param move: the move to be performed. A tuple of the form
            (r coord, q coord, channel).
        :param player: the player performing the move (0 or 1).
        """
        # Place the stones
        for placement in move:
            if placement:  # Must check this because some moves place only one stone
                q, r, c = move
                self.board[c, q, r] = 1

        # Transform the board into a 2D representation
        board_2d = np.copy(self.board)
        for c in range(board_2d.shape[0]):
            board_2d[c, :, :] *= c + 1
        board_2d = np.sum(board_2d, axis=0)

        # Identify blooms
        blooms = []
        for r in range(board_2d.shape[1]):
            for q in range(board_2d.shape[0]):
                if self.is_valid_space((q, r)):
                    # (q, r) is a valid space
                    if board_2d[r, q] > 0:
                        if not any(((q, r) in bloom for bloom in blooms)):
                            bloom = [(q, r)]


        # Remove any fenced blooms (and increment the # of captured stones)

    def add_neighbours(self, bloom, colour, position):
        """A recursive function for finding all stones that belong to the same
        bloom as te stone at the given position.
        """
        raise NotImplementedError
