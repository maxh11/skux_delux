from .aiutils import *
from .heuristic_utils import *


def minimax(node, depth, alpha, beta, maximising_player, colour):
    if depth == 0:
        return node.state.heuristic(node.last_colour)
    current_node = node.copy()

    if maximising_player:
        max_eval = -INFINITY
        children = current_node.get_children(opponent(current_node.last_colour))
        sorted(children, key=lambda child: child.value, reverse=True)
        for i in range(len(children)):
            ev = minimax(children[i], depth - 1, alpha, beta, False, colour)
            max_eval = max(max_eval, ev)
            alpha = max(alpha, ev)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = INFINITY
        children = current_node.get_children(opponent(current_node.last_colour))
        sorted(children, key=lambda child: child.value)
        for i in range(len(children)):
            ev = minimax(children[i], depth - 1, alpha, beta, True, colour)
            min_eval = min(min_eval, ev)
            beta = min(beta, ev)
            if beta <= alpha:
                break
        return min_eval


def alpha_beta_cutoff_search(state, game, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""
    player = game.to_move

    # Functions used by alpha_beta
    def max_value(state_4_max, alpha, beta, depth):
        if cutoff_test(state_4_max, depth):
            return eval_fn(state_4_max)
        v = INFINITY
        posible_actions = game.actions(state_4_max)
        for a in posible_actions:
            v = max(v, min_value(game.result(state_4_max.copy(), a), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state_4_min, alpha, beta, depth):
        if cutoff_test(state_4_min, depth):
            return eval_fn(state_4_min)
        v = INFINITY
        possible_actions = game.actions(state_4_min)
        for a in possible_actions:
            v = min(v, max_value(game.result(state_4_min.copy(), a), alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    cutoff_test = (cutoff_test or (lambda state_lambda, depth: depth > d))
    eval_fn = eval_fn or (lambda state_lambda: game.utility(state_lambda, player))
    best_score = -INFINITY
    beta = INFINITY
    best_action = None
    print(state.total_black())
    posible_actions = game.actions(state)
    for a in posible_actions:
        v = min_value(game.result(state.copy(), a), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action


class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def __init__(self, to_move, state):
        self.to_move = to_move
        self.initial = state

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        # return state.
        return state.get_possible_actions(self.to_move)

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        res = state.copy().apply_action(self.to_move, move)  ## finish this
        self.to_move = opponent(self.to_move)
        return res

    def utility(self, state, player):
        """Return the value of this final state to player."""
        return state.heuristic(player)

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)
