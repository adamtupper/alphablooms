"""A board class for the game of Blooms.
"""


class Board:
    """A board class for the game of Blooms.
    """
    def __init__(self, size=4, score_target=15):
        """Initialise a new game board.

        :param size: the size of the board (either base 4, 5, or 6).
        :param score_target: the number of 'captures' to win the game. It is
            recommended that the number of captures are 15 for a base 4 board,
            20 for a base 5 board, and 25 or 30 for a base 6 board.
        """
        self.score_target = score_target
        self.captures = [0, 0]

    def get_legal_moves(self, player):
        """Returns all the legal moves for the given player.

        :param player: 0 or 1 to denote the player in question.
        :return: the list of all legal moves for the given player.
        """
        raise NotImplementedError

    def has_legal_moves(self):
        raise NotImplementedError

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
