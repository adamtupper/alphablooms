"""Tests for the BloomsLogic module.
"""
from .context import blooms

from itertools import permutations

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


def test_get_board_3d():
    """Check that the the 2D board representation is successfully converted into
    a 3D representation.
    """
    board = Board()

    # Place a few different colour stones on the board
    board.place_stone(position=(3, 0), colour=1)
    board.place_stone(position=(6, 0), colour=2)
    board.place_stone(position=(0, 6), colour=3)
    board.place_stone(position=(3, 6), colour=4)

    board_3d = board.get_board_3d()

    assert board_3d.shape == (4, 7, 7)
    assert np.count_nonzero(board_3d) == 4
    assert board_3d[0, 0, 3] == 1
    assert board_3d[1, 0, 6] == 1
    assert board_3d[2, 6, 0] == 1
    assert board_3d[3, 6, 3] == 1


def test_is_valid_space_invalid():
    """Check that all invalid spaces on a base 4 board are correctly identified
    as invalid.
    """
    board = Board()

    invalid_spaces = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0),
                      (4, 6), (5, 5), (5, 6), (6, 4), (6, 5), (6, 6)]

    for space in invalid_spaces:
        assert not board.is_valid_space(space)


def test_is_valid_space_out_of_range():
    """Check that a position outside the range of the board array is classified
    as invalid.
    """
    board = Board()

    assert not board.is_valid_space((7, 2))


def test_is_valid_space_valid():
    """Check that a valid space on a base 4 board isn't misclassified as an
    invalid space.
    """
    board = Board()

    assert board.is_valid_space((3, 3))


def test_has_legal_moves_some():
    """Check the function correctly returns that there are legal moves when the
    board is not.
    """
    n = 4
    board = Board(size=n)

    # Add a stone to the board for good measure
    board.board_2d[3, 3] = 1

    assert board.has_legal_moves()


def test_has_legal_moves_none():
    """Check the function correctly returns that there are no legal moves when
    the board is full.
    """
    n = 4
    board = Board(size=n)

    # Populate the board with stones of Colour 1
    board.board_2d = np.ones(board.board_2d.shape)

    assert not board.has_legal_moves()


def test_get_legal_moves():
    """Check that the function generates all possible moves when the board is
    empty.
    """
    board = Board()

    empty_spaces = board.get_empty_spaces()
    legal_moves = board.get_legal_moves(player=0)

    for space in empty_spaces:
        q, r = space
        assert [(q, r, 1), ()] in legal_moves
        assert [(q, r, 2), ()] in legal_moves

    for (space1, space2) in permutations(empty_spaces, r=2):
        assert [(space1[0], space1[1], 1), (space2[0], space2[1], 2)] in legal_moves


def test_get_neighbours_center():
    """Check that the correct neighbours are returned for the central position.
    """
    board = Board()

    expected_neighbours = [(3, 2), (4, 2), (4, 3), (3, 4), (2, 4), (2, 3)]
    actual_neighbours = board.get_neighbours(position=(3, 3))

    assert len(actual_neighbours) == len(expected_neighbours)
    assert set(actual_neighbours) == set(expected_neighbours)


def test_get_neighbours_edge():
    """Check that the correct neighbours are returned for an edge position.
    """
    board = Board()

    expected_neighbours = [(3, 6), (3, 5), (4, 4), (5, 4)]
    actual_neighbours = board.get_neighbours(position=(4, 5))

    assert len(actual_neighbours) == len(expected_neighbours)
    assert set(actual_neighbours) == set(expected_neighbours)


def test_find_bloom_members():
    """Check that the function correctly identifies all members of a bloom.
    """
    board = Board()

    # Add a bloom
    board.place_stone(position=(3, 2), colour=1)
    board.place_stone(position=(3, 3), colour=1)
    board.place_stone(position=(3, 4), colour=1)
    board.place_stone(position=(4, 3), colour=1)

    # Add an additional neighbouring stone of a different colour (that is not
    # part of the bloom).
    board.place_stone(position=(4, 2), colour=2)

    expected_bloom = [(3, 2), (3, 3), (3, 4), (4, 3)]
    actual_bloom = board.find_bloom_members(bloom=set(), colour=1, position=(3, 2))

    assert len(actual_bloom) == len(expected_bloom)
    assert set(actual_bloom) == set(expected_bloom)


def test_is_empty_space_empty():
    """Check that empty spaces are correctly identified.
    """
    board = Board()
    assert board.is_empty_space(position=(3, 3))


def test_is_empty_space_filled():
    """Check that empty spaces are correctly identified.
    """
    board = Board()
    board.place_stone(position=(3, 3), colour=1)

    assert not board.is_empty_space(position=(3, 3))


def test_remove_stone_valid():
    """Check that the function removes a stone from the board (when there is
    one present.
    """
    board = Board()
    board.place_stone(position=(3, 3), colour=1)
    board.remove_stone(position=(3, 3))

    assert board.is_empty_space(position=(3, 3))


def test_remove_stone_invalid():
    """Check that the function raises an error when there is no stone to be
    removed.
    """
    board = Board()

    with pytest.raises(AssertionError):
        board.remove_stone(position=(3, 3))


def test_is_fenced_middle():
    """Check that the function correctly classifies a fenced bloom in the middle
    of the board.
    """
    board = Board()

    # Add a bloom
    bloom = [(3, 2), (3, 3), (3, 4), (4, 3)]
    for position in bloom:
        board.place_stone(position, colour=1)

    # Fence the bloom
    for position in bloom:
        neighbours = board.get_neighbours(position)
        for neighbour in neighbours:
            if board.is_empty_space(neighbour):
                board.place_stone(neighbour, colour=2)

    assert board.is_fenced(bloom)


def test_is_fenced_edge():
    """Check that the function correctly classifies a fenced bloom at the edge
    of the board.
    """
    board = Board()

    # Add a bloom
    bloom = [(6, 3), (5, 4)]
    for position in bloom:
        board.place_stone(position, colour=1)

    # Fence the bloom
    for position in bloom:
        neighbours = board.get_neighbours(position)
        for neighbour in neighbours:
            if board.is_empty_space(neighbour):
                board.place_stone(neighbour, colour=2)

    assert board.is_fenced(bloom)


def test_is_fenced_false():
    """Check that the function correctly classifies a non-fenced bloom.
    """
    board = Board()

    # Add a bloom
    bloom = [(3, 2), (3, 3), (3, 4), (4, 3)]
    for position in bloom:
        board.place_stone(position, colour=1)

    # Fence the bloom
    for position in bloom:
        neighbours = board.get_neighbours(position)
        for neighbour in neighbours:
            if board.is_empty_space(neighbour):
                board.place_stone(neighbour, colour=2)

    # Remove one of the stones fencing in the bloom
    board.remove_stone(position=(4, 4))

    assert not board.is_fenced(bloom)
