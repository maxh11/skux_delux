from .alpha_beta_utils import *


def get_greedy_action(colour, base_node):
    """returns the action associated with the best score achieved after that action is enacted on our current_node.state
    Note: because python uses a min-heap for their priority queue implementation, better scores are lower, the lower the score the better its value"""

    # store actions in a dict {} where the key is the score achieved by that action, break ties randomly
    # get the possible actions from our current position

    # make a copy of the initial node we were given
    # base_node = Node(current_node.state)
    best_actions = []  # initialise the best_actions with a dummy value so our loop doesnt kick up a fuss when we try to access the [0] index for the first time
    best_score = LOST_GAME
    actions = base_node.state.get_possible_actions(colour)
    for action in actions:
        current_node = base_node.apply_action(colour, action)
        current_node.value -= manhattan_dist(colour, current_node.state)
        if current_node.value > best_score:
            best_actions = [action]  # reset the list to have 1 action as the new best
            best_score = current_node.value
        elif current_node.value == best_score:
            best_actions.append(action)

    # find the best action in those actions and return it. Break draws randomly
    return random.choice(best_actions)


def get_alphabeta_action(colour, node, budget):
    current_node = node.copy()
    current_game = Game(colour, current_node.state)
    return alpha_beta_cutoff_search(current_node.state, current_game, d=4)

    current_node = node.copy()
    first_moves = current_node.get_children(colour)

    moves = {}
    for i in range(len(first_moves)):
        child = first_moves[i]
        value = minimax(child, budget, -INFINITY, INFINITY, False, colour)
        if value in moves:
            moves[value].append(child.move)
        else:
            moves[value] = [child.move]
    return random.choice(moves[max(moves)])
