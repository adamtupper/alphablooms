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
                if self.is_valid_space((q, r)) and self.is_empty_space((q, r)):
                    # (q, r) is a valid space and is empty
                    empty_spaces.append((q, r))

        return empty_spaces

    def is_valid_space(self, position):
        """Check to see if the given position is a valid space on the board.

        Because of the hexagonal shape of the board, some elements of the 2D
        board representation are not spaces on the board.

        :param position: A tuple representing the (q, r) coord to place the
            stone.
        :return: True if the given position is a valid space, False otherwise.
        """
        q, r = position
        q_in_range = 0 <= q < 2 * self.size - 1
        r_in_range = 0 <= r < 2 * self.size - 1
        not_in_top_left = q + r >= self.size - 1
        not_in_bottom_right = 4 * self.size - 4 - q - r >= self.size - 1

        return q_in_range and r_in_range and not_in_top_left and not_in_bottom_right

    def is_empty_space(self, position):
        """Check to see if a space is empty.

        :param position: A tuple representing the (q, r) coord to place the
            stone.
        :return: True if the given position is empty, False otherwise.
        """
        q, r = position
        return self.board_2d[r, q] == 0

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

    def remove_stone(self, position):
        """Remove a stone from the board.

        :param position: A tuple representing the (q, r) coord to place the
            stone.
        """
        q, r = position

        # Check that there is a stone at the given position
        assert not self.is_empty_space(position)

        self.board_2d[r, q] = 0

    def get_legal_moves(self, player):
        """Returns all the legal moves for the given player.

        Each turn, the player can place up to two stones on any empty spaces on
        the board. However, if the player places two stones they must be
        different colors.

        :param player: 0 or 1 to denote the player in question.
        :return: the list of all legal moves for the given player.
        """
        colour1, colour2 = self.colours[player]
        empty_spaces = self.get_empty_spaces()
        moves = []

        # Add all possible one stone moves of the player's 1st colour
        moves += [[(q, r, colour1), ()] for (q, r) in empty_spaces]

        # Add all possible one stone moves of the player's 2nd colour
        moves += [[(q, r, colour2), ()] for (q, r) in empty_spaces]

        # Generate all possible two stone moves
        moves += [[(m1[0], m1[1], colour1), (m2[0], m2[1], colour2)] for (m1, m2) in permutations(empty_spaces, r=2)]

        return moves

    def has_legal_moves(self):
        """Return True or False depending on whether there are any legal moves
        remaining.

        There are legal moves remaining if there are empty spaces on the board.

        :return: True if there are legal moves, False otherwise.
        """
        # Check each index (q, r) to see if it is empty.
        for r in range(self.board_2d.shape[0]):
            for q in range(self.board_2d.shape[1]):
                if self.is_valid_space((q, r)) and self.board_2d[r][q] == 0:
                    # (q, r) is a valid space and is empty
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
                q, r, colour = move
                self.board_2d[r, q] = colour

        # Identify blooms
        blooms = []
        for r in range(self.board_2d.shape[1]):
            for q in range(self.board_2d.shape[0]):
                if self.is_valid_space((q, r)) and self.board_2d[r, q] > 0:
                    # (q, r) is a valid, non-empty space
                    if not any(((q, r) in bloom for bloom in blooms)):
                        # If the stone is not a member of a currently known bloom
                        colour = self.board_2d[r, q]
                        bloom = self.find_bloom_members({}, colour, (q, r))
                        blooms.append(bloom)

        # Remove any fenced blooms (and increment the # of captured stones)
        for bloom in blooms:
            if self.is_fenced(bloom):
                bloom = list(bloom)

                # TODO: Remove bloom from the board

                # Update captures
                bloom_colour = self.board_2d[bloom[0][1], bloom[0][0]]
                if bloom_colour in (1, 2):
                    self.captures[0] += len(bloom)
                else:
                    self.captures[1] += len(bloom)

    def is_fenced(self, bloom):
        """Check to see if the given bloom is fenced.

        :param bloom: A list of the positions that make up the bloom.
        :return: True if the bloom is fenced, False otherwise.
        """
        for position in bloom:
            for q, r in self.get_neighbours(position):
                if self.board_2d[r, q] == 0:
                    # A neighbouring position is empty
                    return False

        return True

    def get_neighbours(self, position):
        """Return a list of the neighbouring positions to the given position.

        :param position: A tuple representing the (q, r) coord to place the
            stone.
        :return: A list of the neighbouring positions.
        """
        q, r = position
        axial_directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

        neighbours = []
        for dq, dr in axial_directions:
            neighbour = (q + dq, r + dr)
            if self.is_valid_space(neighbour):
                neighbours.append(neighbour)

        return neighbours

    def find_bloom_members(self, bloom, colour, position):
        """A recursive function for finding all stones that belong to the same
        bloom as te stone at the given position.

        :param bloom: A set of all stones in the bloom (on the first call this
            is empty).
        :param colour: The colour of the bloom.
        :param position: The position to start the search for other bloom
            members from.
        :return: The set of all positions with a stone in the bloom.
        """
        neighbours = self.get_neighbours(position)
        neighbours = {n for n in neighbours if self.board_2d[n[1], n[0]] == colour and n not in bloom}

        if not neighbours:
            return bloom
        else:
            bloom |= neighbours
            for neighbour in neighbours:
                bloom |= self.find_bloom_members(bloom, colour, neighbour)
            return bloom
