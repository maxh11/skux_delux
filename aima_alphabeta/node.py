from .heuristic_utils import *
from .aiutils import *
from .state import *


class Node:
    """Node class for storing states and their values"""

    def __init__(self, state=None, value=0, parent=None, move=None, depth=0):
        if state is None:
            self.state = State()
            self.last_colour = BLACK
        else:
            self.state = State(state.white_stacks, state.black_stacks)
        self.value = value
        self.parent = parent
        self.move = move
        self.depth = depth
        self.last_colour = None

    # these functions are for comparing the scores (or 'value') associated with each node
    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

    def __le__(self, other):
        return self.value <= other.value

    def __ge__(self, other):
        return self.value >= other.value

    # this is for checking if the node is already in the explored_nodes array (i.e. it's has the same state)
    # For example, when someone checks if node1 == node2, it will return true if they have the same state
    def __eq__(self, other):
        return self.state == other.state

    def copy(self):
        copy_node = Node(self.state, self.value, self.parent, self.move, self.depth)
        copy_node.last_colour = self.last_colour
        return copy_node

    def apply_action(self, colour, action):
        """return a copy of the base_node after action has happened on it"""
        if action[0] == MOVE:
            return self.move_action(colour, action[1], action[2], action[3])
        if action[0] == BOOM:
            return self.boom_action(colour, action[1])
        # else invalid action
        print("INVALID ACTION GIVEN TO .apply_action() in aiutils.py\n")
        return None

    def get_children(self, colour):
        children = []
        actions = self.state.get_possible_actions(colour)
        for action in actions:
            children.append(self.apply_action(colour, action))
        return children

    def boom_action(self, colour, stack_to_boom):
        # make a new node that is a copy of the base_node
        new_node = self.copy()

        # adjust new_node fields according to how the boom change them:
        # parent node of the new_node is the base_node
        new_node.parent = self
        # new_node depth is parent depth + 1
        new_node.depth = self.depth + 1
        # store the move which got us to new_node
        new_node.move = (BOOM, stack_to_boom)
        new_node.last_colour = colour

        # recursive boom at the new_node.state starting at 'stack', this updates the state
        new_node.state = new_node.state.chain_boom(stack_to_boom)

        # update value and return
        new_node.value = new_node.state.heuristic(colour)

        return new_node

    def move_action(self, colour, n_pieces, stack, dest_square):
        """ apply a move action to the given base node by moving n_pieces from stack n_steps in move_direction
                returns new_node resulting from the move """
        # make a new node that is a copy of the base_node. Set last_player to current colour making the move
        new_node = self.copy()
        # adjust new_node fields according to how our move will change them:
        # parent node of the new_node is the base_node
        new_node.parent = self
        # new_node depth is parent depth + 1
        new_node.depth = self.depth + 1
        # store the move which got us to new_node
        new_node.move = (MOVE, n_pieces, stack, dest_square)
        new_node.last_colour = colour

        # execute move on new_node state
        # move the pieces from the stack to a new stack

        new_node.state.get_colour(colour)[stack] -= n_pieces
        if new_node.state.get_colour(colour)[stack] == 0:
            new_node.state.get_colour(colour).pop(stack)
        if dest_square in new_node.state.get_colour(colour):
            # there is already a stack in the square we are moving to, just add number of pieces
            new_node.state.get_colour(colour)[dest_square] += n_pieces
        else:
            # we have to make a new key value pair because we made a new stack
            new_node.state.get_colour(colour)[dest_square] = n_pieces

        # update node value
        new_node.value = new_node.state.heuristic(colour)

        return new_node
