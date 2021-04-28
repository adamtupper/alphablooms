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
game = BloomsGame(size=4, score_target=15)
human = HumanBloomsPlayer(game).play

# WARNING: The chosen agent should match the game size and score target
model = NNet(game)
model.load_checkpoint('./notebooks/checkpoints_run_3', 'checkpoint_1.pth.tar')

args = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts = MCTS(game, model, args)
agent = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

arena = Arena.Arena(agent, human, game)

print(arena.playGames(2, verbose=True, display=False))
