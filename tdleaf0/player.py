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
            #return get_alphabeta_action(self.colour, self.current_node, budget/2)
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
        num_pieces_evs.append(self.current_node.state.total_white() - self.current_node.state.total_black())
        kill_danger_evs.append(self.current_node.state.kill_danger(colour, OUR_TURN))
        manhattan_dist_evs.append(manhattan_dist(self.current_node.state, colour))
        num_groups_evs.append(self.current_node.state.num_groups(colour))
        # Game over, start td leaf lambda learning being white player always
        if is_game_over(self.current_node.state):

            if colour == WHITE and self.current_node.state.total_black() == 0:
                # game won, sn = 1
                sn = 1
            elif colour == WHITE and self.current_node.state.total_white() == 0:
                # game lost, sn = -1
                sn = -1
            elif colour == BLACK and self.current_node.state.total_white() == 0:
                sn = 1
            elif colour == BLACK and self.current_node.state.total_black() == 0:
                sn = -1
            else:
                # game drew, sn = 0
                sn = 0

            print(sn)
            sum1 = 0
            sum2 = 0
            sum3 = 0
            sum4 = 0
            num_pieces_evs.append(sn)
            kill_danger_evs.append(sn)
            manhattan_dist_evs.append(sn)
            num_groups_evs.append(sn)

            for i in (range(len(num_pieces_evs))-1):
                sum1 += num_pieces_evs[i] * (math.tanh(num_pieces_evs[i]) - math.tanh(num_pieces_evs[i+1]))
            w1new = w1 - 0.005 * sum1
            print(w1new)
            parser.set('weights', 'w1', str(w1new))

            for i in (range(len(kill_danger_evs))-1):
                sum2 += kill_danger_evs[i] * (math.tanh(kill_danger_evs[i]) - math.tanh(kill_danger_evs[i+1]))
            w2new = w2 - 0.005 * sum2
            print(w2new)
            parser.set('weights', 'w2', str(w2new))

            for i in (range(len(manhattan_dist_evs))-1):
                sum3 += manhattan_dist_evs[i] * (math.tanh(manhattan_dist_evs[i]) - math.tanh(manhattan_dist_evs[i+1]))
            w3new = w3 + 0.005 * sum3
            print(w3new)
            parser.set('weights', 'w3', str(w3new))

            for i in (range(len(num_groups_evs))-1):
                sum4 += num_groups_evs[i] * (math.tanh(num_groups_evs[i]) - math.tanh(num_groups_evs[i+1]))
            w4new = w4 - 0.005 * sum4
            print(w4new)
            parser.set('weights', 'w4', str(w4new))

            # Writing our configuration file to 'example.ini'
            with open('./tdleaf0/weights.ini', 'w') as configfile:
                parser.write(configfile)
            configfile.close()

        # print({WHITE: heuristic(WHITE, self.current_node.state), BLACK: heuristic(BLACK, self.current_node.state)})