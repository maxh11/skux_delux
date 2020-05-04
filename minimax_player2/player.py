import random
from .aiutils import *

"""Actions syntax:
    ("MOVE", n, (xa, ya), (xb, yb))
    ("BOOM", (x, y))
"""

budget = 2


class MinimaxPlayer:
    def __init__(self, colour):
        """
        This method is called once at the beginning of the game to initialise
        your player. You should use this opportunity to set up your own internal
        representation of the game state, and any other information about the
        game state you would like to maintain for the duration of the game.
        The parameter colour will be a string representing the player your
        program will play as (White or Black). The value will be one of the
        strings "white" or "black" correspondingly.
        """
        # TODO: Set up state representation.

        # set up our current node and
        self.current_node = Node()
        self.colour = colour
        # print({WHITE: heuristic(WHITE, self.current_node.state), BLACK: heuristic(BLACK, self.current_node.state)})

    def action(self):
        """
        This method is called at the beginning of each of your turns to request
        a choice of action from your program.
        Based on the current state of the game, your player should select and
        return an allowed action to play on this turn. The action must be
        represented based on the spec's instructions for representing actions.
        """
        # TODO: Decide what action to take, and return it
        # this player of ours will just pick a random one
        # normally 18 with depth of 2
        if self.current_node.state.total_pieces() > 18:
            return get_greedy_action(self.colour, self.current_node, budget)
        return get_alphabeta_action(self.colour, self.current_node, budget)

    def update(self, colour, action):
        """
        This method is called at the end of every turn (including your player’s
        turns) to inform your player about the most recent action. You should
        use this opportunity to maintain your internal representation of the
        game state and any other information about the game you are storing.
        The parameter colour will be a string representing the player whose turn
        it is (White or Black). The value will be one of the strings "white" or
        "black" correspondingly.
        The parameter action is a representation of the most recent action
        conforming to the spec's instructions for representing actions.
        You may assume that action will always correspond to an allowed action
        for the player colour (your method does not need to validate the action
        against the game rules).
        """
        # TODO: Update state representation in response to action.
        self.current_node = self.current_node.apply_action(colour, action)
        # print({WHITE: heuristic(WHITE, self.current_node.state), BLACK: heuristic(BLACK, self.current_node.state)})