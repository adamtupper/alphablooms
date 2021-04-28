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


class HumanBloomsPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        board.visualise(show_coords=True)
        valid = self.game.getValidMoves(board, 1)

        while True:
            input_move = input('Enter move: ')

            try:
                # Check formatting of input
                input_move = eval(input_move)
                assert len(input_move) == 2
                assert all([type(a) == tuple for a in input_move])
                assert all([type(x) == int for a in input_move for x in a])
                if input_move[1]:
                    assert all([len(a) == 3 for a in input_move])
                else:
                    assert len(input_move[0]) == 3
            except (AssertionError, SyntaxError):
                print('Invalid format for move. Must be formatted ((q, r, c1),(q, r, c2)) or ((q, r, c1), ()) or ((q, r, c2), ()).')
                continue

            try:
                # Check validity of move
                move_idx = board.move_map_player_0[input_move]
                assert valid[move_idx]
            except (KeyError, AssertionError):
                print('Invalid move.')
                continue

            # Move is valid, break from loop
            break

        return move_idx
