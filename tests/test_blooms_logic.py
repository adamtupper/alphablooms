"""Tests for the BloomsLogic module.
"""
from .context import blooms

import numpy as np
import pytest

from blooms.BloomsLogic import Board


def test_dummy():
    pass


def test_board_default():
    """Test the Board constructor with default parameter values.
    """
    board = Board()

    assert board.size == 4
    assert board.score_target == 15
    assert board.captures == [0, 0]
    assert board.colours == [(1, 2), (3, 4)]
    assert np.all(board.board_2d == 0)
    assert board.board_2d.shape == (7, 7)


def test_empty_spaces_empty():
    """Check that the correct number of empty spaces are returned for an empty
    board.
    """
    n = 4
    board = Board(size=n)

    empty_spaces = board.get_empty_spaces()
    assert len(empty_spaces) == 3*n**2 - 3*n + 1


def test_empty_spaces_partial():
    """Check that the correct number of empty spaces are returned for a
    partially filled board.
    """
    n = 4
    board = Board(size=n)

    board.board_2d[3, 3] = 1

    empty_spaces = board.get_empty_spaces()
    assert len(empty_spaces) == 3*n**2 - 3*n


def test_empty_spaces_full():
    """Check that the correct number of empty spaces are returned for a
    full board.
    """
    n = 4
    board = Board(size=n)

    board.board_2d = np.ones(board.board_2d.shape)

    empty_spaces = board.get_empty_spaces()
    assert len(empty_spaces) == 0


def test_place_stone_valid():
    """Check that place_stone correctly updates the board state when the
    placement is valid.
    """
    n = 4
    board = Board(size=n)

    position = (6, 0)
    board.place_stone(position, colour=1)

    assert board.board_2d[position[1], position[0]] == 1
    assert np.count_nonzero(board.board_2d) == 1


def test_place_stone_invalid_non_empty():
    """Check that an error is raised if the position is already filled.
    """
    n = 4
    board = Board(size=n)

    position = (6, 0)
    board.place_stone(position, colour=1)

    with pytest.raises(AssertionError):
        board.place_stone(position, colour=1)


def test_place_stone_invalid_position():
    """Check that an error is raised if the position is invalid (i.e. lies
    outside the board).
    """
    n = 4
    board = Board(size=n)

    with pytest.raises(AssertionError):
        position = (0, 0)
        board.place_stone(position, colour=1)


def test_get_3d_board():
    """Check that the the 2D board representation is successfully converted into
    a 3D representation.
    """
    board = Board()

    # Place a few different colour stones on the board
    board.place_stone(position=(3, 0), colour=1)
    board.place_stone(position=(6, 0), colour=2)
    board.place_stone(position=(0, 6), colour=3)
    board.place_stone(position=(3, 6), colour=4)

    board_3d = board.get_3d_board()

    assert board_3d.shape == (4, 7, 7)
    assert np.count_nonzero(board_3d) == 4
    assert board_3d[0, 0, 3] == 1
    assert board_3d[1, 0, 6] == 1
    assert board_3d[2, 6, 0] == 1
    assert board_3d[3, 6, 3] == 1

