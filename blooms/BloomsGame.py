"""Game class for Blooms.
"""
import sys

sys.path.append('..')

import numpy as np
from scipy.special import perm

from Game import Game

from blooms.BloomsLogic import Board


class BloomsGame(Game):
    """This class specifies the Game class for Blooms.
    """

    def __init__(self, size=4, score_target=15):
        self.size = size
        self.score_target = score_target

    def getInitBoard(self):
        """
        Returns:
            startBoard: The initial game board. Note that this is not the form that
            will be passed to the neural network, but the entire board class.
        """
        board = Board(self.size, self.score_target)
        return board

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return 2 * self.size - 1, 2 * self.size - 1

    def getActionSize(self):
        """Note that function returns the maximum number of possible actions
        (i.e. for when the board is empty), NOT the number of valid actions for
        a particular board state.

        Returns:
            actionSize: number of all possible actions
        """
        n_spaces = (3 * self.size ** 2) - (3 * self.size) + 1
        n_one_stone_moves = 2 * n_spaces
        n_two_stone_moves = perm(n_spaces, 2)

        return int(n_one_stone_moves + n_two_stone_moves)

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: the integer index of the action taken by current player
        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        board = board.copy()

        # Fetch the action that corresponds to the action index
        if player == 1:
            move = board.move_map_player_1.inverse[action]
        else:
            move = board.move_map_player_0.inverse[action]

        if board.is_legal_move(move):
            board.execute_move(move, player)

        return board, -player

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player
        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        player = 0 if player == -1 else 1
        valid_moves = board.get_legal_moves(player)

        valid_moves_vec = np.zeros(int(self.getActionSize()))
        for move in valid_moves:
            if player == 0:
                move_idx = board.move_map_player_0[move]
            else:
                move_idx = board.move_map_player_1[move]

            valid_moves_vec[move_idx] = 1

        return valid_moves_vec

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)
        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        player = 0 if player == -1 else 1
        opponent = 0 if player == 1 else 1

        winning_state = board.is_win(player)
        losing_state = board.is_win(opponent)

        if winning_state:
            return 1.0
        elif losing_state:
            return -1.0
        elif board.has_legal_moves():
            # The game is still ongoing
            return 0.0
        else:
            # The game is a draw
            # TODO: Investigate whether the size of this "small" value matters
            return 1e-4

    def getCanonicalForm(self, board, player):
        """The canonical form of the board is from the POV of Player 1. When
        the player is Player -1, we need to reverse the colours of the stones on
        the board (i.e. 1 <-> 3, 2 <-> 4) and reverse the number of captures.

        Input:
            board: current board
            player: current player (1 or -1)
        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        board = board.copy()

        if player == 1:
            # The board is already in the right POV
            return board
        else:
            # We need to switch the POV
            board.captures.reverse()

            for r in range(board.board_2d.shape[0]):
                for q in range(board.board_2d.shape[1]):
                    if board.board_2d[r, q] == board.colours[0][0]:
                        # Player -1, Colour 1 -> Player 1, Colour 1
                        board.board_2d[r, q] = board.colours[1][0]
                    elif board.board_2d[r, q] == board.colours[0][1]:
                        # Player -1, Colour 2 -> Player 1, Colour 2
                        board.board_2d[r, q] = board.colours[1][1]
                    elif board.board_2d[r, q] == board.colours[1][0]:
                        # Player 1, Colour 1 -> Player -1, Colour 1
                        board.board_2d[r, q] = board.colours[0][0]
                    elif board.board_2d[r, q] == board.colours[1][1]:
                        # Player 1, Colour 1 -> Player -1, Colour 2
                        board.board_2d[r, q] = board.colours[0][1]

            return board

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()
        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        shift = self.size - 1
        transforms = [
            # new x, new y, new z
            (1, 0, 2),
            (2, 1, 0),
            (0, 2, 1)
        ]

        reflected_forms = []
        for t in transforms:
            for multiplier in [-1, 1]:
                for n_rotations in range(0, 6):
                    refl_board = board.copy()
                    refl_pi = pi.copy()

                    for q in range(board.board_2d.shape[0]):
                        for r in range(board.board_2d.shape[0]):
                            if board.is_valid_space(position=(q, r)):
                                # Centre board at the origin and convert position to cube coordinates
                                cube_coord = board.axial_to_cube(q - shift, r - shift)

                                # Apply the rotation (flip signs and shift the components one place to the right for
                                # each 60 degree rotation)
                                rot_cube_coord = np.array(cube_coord)
                                for i in range(n_rotations):
                                    rot_cube_coord = np.roll(-rot_cube_coord, 1)

                                # Apply the reflection
                                refl_cube_coord = [multiplier * c for c in rot_cube_coord]
                                refl_cube_coord = [refl_cube_coord[i] for i in t]

                                # Convert the reflected position to axial coordinates
                                refl_q, refl_r = board.cube_to_axial(*refl_cube_coord)

                                # Shift the board back to being centred at (size, size)
                                refl_q, refl_r = refl_q + shift, refl_r + shift

                                # print(f'Axial coord: {(q, r)} -> Centred axial coord: {(q - shift, r - shift)} -> Cube coord: {cube_coord} -> Refl cube coord: {tuple(refl_cube_coord)} -> Refl centred axial coord: {(refl_q - shift, refl_r - shift)} -> Refl axial coord: {(refl_q, refl_r)}')

                                # Update the reflected position on the reflected board
                                refl_board.board_2d[refl_r, refl_q] = board.board_2d[r, q]

                                # Update policy vector
                                for move, idx in board.move_map_player_0.items():
                                    # It doesn't matter which player we use since we're not interested in the colour
                                    for action in move:
                                        if action and action[0] == q and action[1] == r:
                                            refl_pi[idx] = pi[idx]

                    reflected_forms.append([refl_board, refl_pi])

        return reflected_forms

    def stringRepresentation(self, board):
        """
        Input:
            board: current board
        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.board_2d.tostring() + bytes(board.captures)
