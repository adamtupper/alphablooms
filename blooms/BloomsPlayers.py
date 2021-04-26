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


class GreedyPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, 1)

        candidates = []
        for move, valid in enumerate(valid_moves):
            if valid:
                next_board, _ = self.game.getNextState(board, 1, move)
                candidates.append((-next_board.captures[1], move))
        candidates.sort()

        return candidates[0][1]
