"""Players class for Blooms.
"""
import numpy as np


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, 1)

        action = np.random.randint(self.game.getActionSize())
        while valid_moves[action] != 1:
            action = np.random.randint(self.game.getActionSize())

        return action
