"""Use this script to play manually against a Blooms agent.
"""
import numpy as np

import Arena
from MCTS import MCTS
from blooms.BloomsGame import BloomsGame
from blooms.BloomsPlayers import *
from blooms.pytorch.NNet import NNetWrapper as NNet
from utils import *

# WARNING: The game size and score target should match the chosen agent
game = BloomsGame(size=5, score_target=20)
human = HumanBloomsPlayer(game).play

# WARNING: The chosen agent should match the game size and score target
model = NNet(game)
model.load_checkpoint('./notebooks/results/chkpts_board5_24hrs', 'best.pth.tar')

args = dotdict({'numMCTSSims': 100, 'cpuct':1.0})
mcts = MCTS(game, model, args)
agent = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

arena = Arena.Arena(agent, human, game)

print(arena.playGames(2, verbose=True, display=False))
