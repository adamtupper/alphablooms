"""Game class for Blooms.
"""
import numpy as np
from scipy.special import perm

from blooms.Game import Game  # TODO: Replace with AlphaZero General import

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
        n_spaces = (3 * self.size**2) - (3 * self.size) + 1
        n_one_stone_moves = 2 * n_spaces
        n_two_stone_moves = perm(n_spaces, 2)

        return n_one_stone_moves + n_two_stone_moves

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player
        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        if board.is_legal_move(action):
            board.execute_move(action, player)

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
        pass

    def getCanonicalForm(self, board, player):
        """
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
        pass

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
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board
        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass