""""

    This is a Regression Test Suite to automatically test all combinations of games and ML frameworks. Each test
    plays two quick games using an untrained neural network (randomly initialized) against a random player.

    In order for the entire test suite to run successfully, all the required libraries must be installed.  They are:
    Pytorch, Keras, Tensorflow.

     [ Games ]      Pytorch     Tensorflow  Keras
      -----------   -------     ----------  -----
    - Othello       [Yes]       [Yes]       [Yes]
    - TicTacToe                             [Yes]
    - Connect4                  [Yes]
    - Gobang                    [Yes]       [Yes]

"""

import unittest

import Arena
from MCTS import MCTS

from blooms.BloomsGame import BloomsGame
from blooms.BloomsPlayers import RandomPlayer, GreedyPlayer
from blooms.pytorch.NNet import NNetWrapper as BloomsPyTorchNNet

import numpy as np
from utils import *

class TestAllGames(unittest.TestCase):

    @staticmethod
    def execute_game_test_random(game, neural_net):
        rp = RandomPlayer(game).play

        args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
        mcts = MCTS(game, neural_net(game), args)
        n1p = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

        arena = Arena.Arena(n1p, rp, game)
        print('Random Opponent...')
        wins, losses, draws = arena.playGames(2, verbose=True, display=False)
        print(f'Wins = {wins}')
        print(f'Losses = {losses}')
        print(f'Draws = {draws}')

    @staticmethod
    def execute_game_test_greedy(game, neural_net):
        gp = GreedyPlayer(game).play

        args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
        mcts = MCTS(game, neural_net(game), args)
        n1p = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

        arena = Arena.Arena(n1p, gp, game)
        print('Greedy Opponent...')
        wins, losses, draws = arena.playGames(2, verbose=True, display=False)
        print(f'Wins = {wins}')
        print(f'Losses = {losses}')
        print(f'Draws = {draws}')

    def test_blooms_pytorch(self):
        self.execute_game_test_random(BloomsGame(), BloomsPyTorchNNet)
        # self.execute_game_test_greedy(BloomsGame(), BloomsPyTorchNNet)


if __name__ == '__main__':
    unittest.main()
