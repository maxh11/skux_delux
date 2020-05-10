from .aiutils import *
from .heuristic_utils import *


class State:
    """State class to be associated with 1  The state is stored in the form of 2 dictionaries, one for white stacks
    and one for black stacks. Each dict has keys which are the (x, y) coordinates of a square where there is at least 1
    piece. The values associated with each x, y coordinate key is the integer number of pieces in that square.
    """

    def __init__(self, white_stacks=None, black_stacks=None):
        # e.g. white_stacks = {(3,2): 1, (3,4): 3}
        if white_stacks is None:
            self.white_stacks = dict(START_WHITE_STACKS)
        else:
            self.white_stacks = dict(white_stacks)
        if black_stacks is None:
            self.black_stacks = dict(START_BLACK_STACKS)
        else:
            self.black_stacks = dict(black_stacks)

    def __eq__(self, other):
        return bool((self.white_stacks == other.white_stacks) and (self.black_stacks == other.black_stacks))

    def total_white(self):
        return sum(self.white_stacks.values())

    def total_black(self):
        return sum(self.black_stacks.values())

    def total_pieces(self, colour=None):
        if colour is None:
            return self.total_black() + self.total_white()
        if colour == WHITE:
            return self.total_white()
        return self.total_black()

    def get_colour(self, colour):
        """returns the dictionary of stacks for colour='white' or colour='black' """
        if colour == WHITE:
            return self.white_stacks
        if colour == BLACK:
            return self.black_stacks
        # else, invalid colour given so return None
        return None

    def get_squares(self, colour):
        """returns the keys of the stack dictionary for 'colour'='white' or 'colour'='black'.
        Since the keys of the stack dictionaries are the (x, y) coordinates where there are at least 1 piece,
        the return type is a list of (x, y) tuples where there are 'colour' pieces"""
        return list(self.get_colour(colour).keys())

    def get_possible_actions(self, colour):
        """Return a list of legal actions for 'colour' from the current node's state E.g. could return something like
        [(BOOM, (0, 2)), (BOOM, (0, 1)), (MOVE, 2, (0, 1), (2, 1)), (MOVE, 1, (7, 5), (7, 6)) ... etc]
        """
        # array of actions which we will return after we have filled it with possible (legal) moves
        actions = []

        # go through the list of applicable BOOM actions and add them to actions[]
        squares = self.get_squares(colour)
        for stack in squares:
            actions.append((BOOM, stack))

        # go through the list of applicable MOVE actions
        # for each item from .items() from a stack dictionary, item[0] is the (x, y) coordinates of of the stack and
        # item[1] is the number of pieces in the stack
        for stack in self.get_colour(colour).items():
            # iterate through each possible number of pieces to move from our stack at the current occupied_square
            for n_pieces in range(1, stack[1] + 1):
                # possible moving directions
                for move_direction in MOVE_DIRECTIONS:
                    # number of squares to move n_pieces from current stack, 1 <= n_steps <= n_pieces
                    for n_steps in range(1, stack[1] + 1):
                        # check if moving n_steps in move_direction from current stack is a legal move (i.e. not out of
                        # bounds and not landing on an enemy piece)
                        if is_legal_move(self.get_squares(opponent(colour)), stack[0], move_direction, n_steps):
                            final_square = calculate_dest_square(stack[0], move_direction, n_steps)
                            actions.append((MOVE, n_pieces, stack[0], final_square))
        return actions

    def get_nontrivial_boom_actions(self, colour):
        # dont return square that has the same resulting state
        # dont return a boom action that only removes our pieces
        current = self.copy()
        actions = []
        while len(current.get_squares(colour)) > 0:
            stack_to_boom = current.get_squares(colour)[0]
            actions.append((BOOM, stack_to_boom))
            current = current.chain_boom(stack_to_boom)
        return actions

    def num_groups(self, colour):
        """A group is defined as a set of pieces over one or more squares which would all be removed in
        a boom action from a boom in any of the pieces in that group.
        That is, a BOOM of any stack in a group of stacks will BOOM all other stacks in that group"""
        # the number of distinct groups is equal to the number of possible boom actions until we have no more stacks left
        return len(self.get_nontrivial_boom_actions(colour))

    def chain_boom(self, stack_to_boom, stacks_to_remove=None):

        # add the stack_to_boom to the stacks_to_remove
        if stacks_to_remove is None:
            stacks_to_remove = set()
        stacks_to_remove.add(stack_to_boom)

        # go through all the adjacent stacks to stack_to_boom
        # add the stack to the stacks to be removed
        # make a boom radius from stack_to_boom
        radius_x = [stack_to_boom[0], stack_to_boom[0], stack_to_boom[0] - 1, stack_to_boom[0] - 1,
                    stack_to_boom[0] - 1, stack_to_boom[0] + 1,
                    stack_to_boom[0] + 1, stack_to_boom[0] + 1]
        # possible corresponding y coordinates e.g. (2,2): [1, 3, 2, 1, 3, 2, 1, 3]
        radius_y = [stack_to_boom[1] - 1, stack_to_boom[1] + 1, stack_to_boom[1], stack_to_boom[1] - 1,
                    stack_to_boom[1] + 1, stack_to_boom[1],
                    stack_to_boom[1] - 1, stack_to_boom[1] + 1]
        radius = list(zip(radius_x, radius_y))

        # get a list of all the squares where the boom hit
        all_stacks = list(self.white_stacks.keys()) + list(self.black_stacks.keys())
        stacks_hit = list(set(all_stacks).intersection(radius))

        # add all the stacks_stacks_hit to the stacks_to_remove set, if they havent been added before, boom them
        for st in stacks_hit:
            if st not in stacks_to_remove:
                stacks_to_remove.add(st)
                self.chain_boom(st, stacks_to_remove)

        # remove stacks_to_remove from state and return state
        for st in stacks_to_remove:
            if st in self.white_stacks:
                self.white_stacks.pop(st)
            if st in self.black_stacks:
                self.black_stacks.pop(st)
        return self

    def heuristic(self, colour):
        if (self.total_black() == 0) and (self.total_white() == 0):
            return DRAW_GAME

        if colour == WHITE:
            if self.total_black() == 0:
                # win game
                return WIN_GAME
            if self.total_white() == 0:
                # lost game
                return LOST_GAME
            # else, the heuristic is the number of our pieces on the board - enemy pieces on the board + manhattan
            # distance, **higher** is better
            return self.total_white() - self.total_black() + 0.1 * self.num_groups(WHITE)

        if colour == BLACK:
            if self.total_white() == 0:
                # win game
                return WIN_GAME
            if self.total_black() == 0:
                # lost game
                return LOST_GAME
            # else, the heuristic is the number of our pieces on the board - enemy pieces on the board + manhattan
            # distance, **higher** is better
            return self.total_black() - self.total_white() + 0.1 * self.num_groups(BLACK)

    def apply_action(self, colour, action):
        """return a copy of the base_node after action has happened on it"""
        if action[0] == MOVE:
            return self.move_action(colour, action[1], action[2], action[3])
        if action[0] == BOOM:
            return self.boom_action(colour, action[1])
        # else invalid action
        print("INVALID ACTION GIVEN TO .apply_action() in state.py\n")
        return None

    def move_action(self, colour, n_pieces, stack, dest_square):
        new_state = self.copy()

        new_state.get_colour(colour)[stack] -= n_pieces
        if new_state.get_colour(colour)[stack] == 0:
            new_state.get_colour(colour).pop(stack)
        if dest_square in new_state.get_colour(colour):
            # there is already a stack in the square we are moving to, just add number of pieces
            new_state.get_colour(colour)[dest_square] += n_pieces
        else:
            # we have to make a new key value pair because we made a new stack
            new_state.get_colour(colour)[dest_square] = n_pieces

    def boom_action(self, colour, stack_to_boom):
        new_state = self.copy()
        return new_state.chain_boom(stack_to_boom)

    def copy(self):
        return State(self.white_stacks, self.black_stacks)
