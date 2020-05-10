import random
import sys

# values for the heuristic
LEFT = (-1, 0)
RIGHT = (1, 0)
UP = (0, 1)
DOWN = (0, -1)
MOVE_DIRECTIONS = [LEFT, RIGHT, UP, DOWN]

# values for the heuristic
INFINITY = sys.maxsize
LOST_GAME = -INFINITY
WIN_GAME = INFINITY
DRAW_GAME = LOST_GAME / 2

MOVE = "MOVE"
BOOM = "BOOM"
WHITE = "white"
BLACK = "black"

START_BLACK_STACKS = {
    (0, 7): 1, (1, 7): 1, (3, 7): 1, (4, 7): 1, (6, 7): 1, (7, 7): 1,
    (0, 6): 1, (1, 6): 1, (3, 6): 1, (4, 6): 1, (6, 6): 1, (7, 6): 1}
START_WHITE_STACKS = {
    (0, 1): 1, (1, 1): 1, (3, 1): 1, (4, 1): 1, (6, 1): 1, (7, 1): 1,
    (0, 0): 1, (1, 0): 1, (3, 0): 1, (4, 0): 1, (6, 0): 1, (7, 0): 1}

OUR_TURN = "ours"
THEIR_TURN = "theirs"

MANHATTAN = "md"
NUM_GROUPS = "ng"


def is_legal_move(enemy_stack_locations, moving_stack_location, move_direction, n_steps):
    """ check if moving n_steps in move_direction from current stack is a legal move (i.e. not out of bounds and not
    landing on an enemy piece) """
    dest_square = calculate_dest_square(moving_stack_location, move_direction, n_steps)
    return bool((dest_square[0] in range(0, 8)) and (dest_square[1] in range(0, 8)) and (
            dest_square not in enemy_stack_locations))


def calculate_dest_square(moving_stack_location, move_direction, n_steps):
    return (
        moving_stack_location[0] + n_steps * move_direction[0], moving_stack_location[1] + n_steps * move_direction[1])


def manhattan_dist(colour, state):
    total = 0

    if colour == WHITE:
        for white in state.white_stacks.items():
            current_total = 0
            for black in state.black_stacks.items():
                current_total += (abs(white[0][0] - black[0][0]) + abs(white[0][1] - black[0][1]))
            if len(state.black_stacks) < 3:
                total += current_total / 5
            else:
                total += current_total / len(state.black_stacks)
        # if len(state.black_stacks) == 0: return total
        if len(state.white_stacks) < 3:
            return total / 5
        return total / len(state.white_stacks)

    if colour == BLACK:
        for black in state.black_stacks.items():
            current_total = 0
            for white in state.white_stacks.items():
                current_total += (abs(white[0][0] - black[0][0]) + abs(white[0][1] - black[0][1]))
            if len(state.white_stacks) < 3:
                total += current_total / 5
            else:
                total += current_total / len(state.white_stacks)
        # if len(state.black_stacks) == 0: return total
        if len(state.black_stacks) < 3:
            return total / 5
        return total / len(state.black_stacks)


def is_game_over(state):
    return bool(state.total_black() == 0 or state.total_white() == 0)


def opponent(colour):
    """get the opponent colour. If given 'white' return 'black' and vice versa"""
    if colour == WHITE:
        return BLACK
    if colour == BLACK:
        return WHITE
    return None
