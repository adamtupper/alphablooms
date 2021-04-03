"""Test for the BloomsGame module.
"""
from .context import blooms

import numpy as np
import pytest
from scipy.special import perm

from blooms.BloomsGame import BloomsGame
from blooms.BloomsGame import Board


def test_init_default():
    """Check that a new default game is initialised correctly.
    """
    game = BloomsGame()

    assert game.size == 4
    assert game.score_target == 15


def test_init_custom():
    """Check that a new custom game is initialised correctly.
    """
    game = BloomsGame(size=5, score_target=20)

    assert game.size == 5
    assert game.score_target == 20


def test_get_init_board():
    """Check that a new board is initialised correctly.
    """
    game = BloomsGame()
    board = game.getInitBoard()

    assert type(board) == Board


def test_get_board_size():
    """Check that the correct board size is returned.
    """
    game = BloomsGame()

    assert game.getBoardSize() == (7, 7)


def test_get_action_size_base4():
    """Check that the correct action space size is returned for a base 4 board.
    """
    game = BloomsGame(size=4)
    n_spaces = 37

    assert game.getActionSize() == 2 * n_spaces + perm(n_spaces, 2)


def test_get_action_size_base3():
    """Check that the correct action space size is returned for a base 3 board.
    """
    game = BloomsGame(size=3)
    n_spaces = 19

    assert game.getActionSize() == 2 * n_spaces + perm(n_spaces, 2)


def test_get_next_state_invalid_move():
    """Check that the original (i.e. unchanged) board state is returned if an
    invalid move is performed.
    """
    game = BloomsGame(size=4)
    board = game.getInitBoard()

    # Initialise the board with a stone
    board.place_stone((6, 2), colour=1)

    initial_board_array = board.get_board_3d()

    # Attempt an illegal move
    player = 1
    move = [(6, 2, 2), ()]
    next_board, next_player = game.getNextState(board, player=1, action=move)

    post_move_board_array = next_board.get_board_3d()

    assert next_player == -player
    assert np.all(initial_board_array == post_move_board_array)


def test_get_next_state_valid_move():
    """Check that the board state is updated if a legal move is performed.
    """
    game = BloomsGame(size=4)
    board = game.getInitBoard()

    # Attempt a legal move
    player = 1
    move = [(6, 2, 2), ()]
    next_board, next_player = game.getNextState(board, player, action=move)

    next_board.board_2d
    for r in range(next_board.board_2d.shape[0]):
        for q in range(next_board.board_2d.shape[1]):
            if (q == 6) and (r == 2):
                assert next_board.board_2d[r, q] == 2
            else:
                assert next_board.board_2d[r, q] == 0

    assert next_player == -player


def test_get_valid_moves():
    """Test that the valid moves vector is correctly built.
    """
    game = BloomsGame(size=4)
    board = game.getInitBoard()

    valid_moves = game.getValidMoves(board, player=-1)
    assert len(valid_moves) == game.getActionSize()
    assert np.all((valid_moves == 0) | (valid_moves == 1))


def test_get_game_ended_win():
    """Check that the function correctly signals a win.
    """
    game = BloomsGame(size=4, score_target=15)
    board = game.getInitBoard()

    # Set the number of captures for Player -1/0 to 15
    board.captures[0] = 15

    assert game.getGameEnded(board, player=-1) == 1.0


def test_get_game_ended_loss():
    """Check that the function correctly signals a loss.
    """
    game = BloomsGame(size=4, score_target=15)
    board = game.getInitBoard()

    # Set the number of captures for Player 1 (i.e. the opponent) to 15
    board.captures[1] = 15

    assert game.getGameEnded(board, player=-1) == -1.0


def test_get_game_ended_ongoing():
    """Check that the function correctly signals an ongoing game.
    """
    game = BloomsGame(size=4, score_target=15)
    board = game.getInitBoard()

    assert game.getGameEnded(board, player=-1) == 0.0


def test_get_game_ended_draw():
    """Check that the function correctly signals a draw.
    """
    game = BloomsGame(size=4, score_target=15)
    board = game.getInitBoard()

    # Fill the board
    for position in board.get_empty_spaces():
        board.place_stone(position=position, colour=1)

    assert game.getGameEnded(board, player=-1) == pytest.approx(1e-4)


def test_get_canonical_form_no_inversion():
    """Check that the original board is returned when called for Player -1.
    """
    game = BloomsGame(size=4, score_target=15)
    board = game.getInitBoard()

    # Place stones
    board.place_stone(position=(5, 1), colour=1)
    board.place_stone(position=(5, 2), colour=2)
    board.place_stone(position=(1, 4), colour=3)
    board.place_stone(position=(1, 5), colour=4)

    # Update captures
    board.captures = [5, 10]

    board = game.getCanonicalForm(board, player=-1)

    assert board.board_2d[1, 5] == 1
    assert board.board_2d[2, 5] == 2
    assert board.board_2d[4, 1] == 3
    assert board.board_2d[5, 1] == 4

    assert board.captures == [5, 10]


def test_get_canonical_form_inversion():
    """Check that the board is inverted (i.e. Player -1 becomes Player 1 and
    vice-a-versa) when called for Player 1.
    """
    game = BloomsGame(size=4, score_target=15)
    board = game.getInitBoard()

    # Place stones
    board.place_stone(position=(5, 1), colour=1)
    board.place_stone(position=(5, 2), colour=2)
    board.place_stone(position=(1, 4), colour=3)
    board.place_stone(position=(1, 5), colour=4)

    # Update captures
    board.captures = [5, 10]

    # Uncomment to visualise the inversion
    # board.visualise()

    board = game.getCanonicalForm(board, player=1)

    assert board.board_2d[1, 5] == 3
    assert board.board_2d[2, 5] == 4
    assert board.board_2d[4, 1] == 1
    assert board.board_2d[5, 1] == 2

    assert board.captures == [10, 5]

    # Uncomment to visualise the inversion
    # board.visualise()
