from sys import maxsize
import random
import sys
from .aiutils import *


# tree builder, partly inspired by Trevor Payne (https://www.youtube.com/watch?v=fInYh90YMJU)
class MMnode:
    def __init__(self, depth, player_to_move, state, value=0):
        self.depth = depth
        self.player_to_move = player_to_move
        self.state = state
        self.children = get_child_nodes()
        self.create_children()

    def create_children(self):
        if self.depth == 0:
            for i in range(1, 3):
                print(hi)

    def get_child_nodes(self, player_to_move=opponent(self.player)):
        if player_to_move is Node:

        """return an array of child nodes. That is, all the possible nodes resulting from legal actions from the current node"""
        legal_actions = get_possible_actions(self.state, opponent(self.last_player))
        children = []
        for action in legal_actions:
            children.append(apply_action(self, action, opponent(self.last_player)))
        return children

    def get_possible_actions(self, colour):
        """Return a list of legal actions for 'colour' from the current node's state E.g. could return something like
        [(BOOM, (0, 2)), (BOOM, (0, 1)), (MOVE, 2, (0, 1), (2, 1)), (MOVE, 1, (7, 5), (7, 6)) ... etc]
        """
        return get_possible_actions(self.state, colour)
