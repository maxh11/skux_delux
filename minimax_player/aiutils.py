import random
import sys

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

    def copy(self):
        return State(self.white_stacks, self.black_stacks)



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
            return move_action(colour, self, action[1], action[2], action[3])
        if action[0] == BOOM:
            return boom_action(colour, self, action[1])
        # else invalid action
        print("INVALID ACTION GIVEN TO .apply_action() in aiutils.py\n")
        return None

    def get_possible_actions(self, colour):
        """Return a list of legal actions for 'colour' from the current node's state E.g. could return something like
        [(BOOM, (0, 2)), (BOOM, (0, 1)), (MOVE, 2, (0, 1), (2, 1)), (MOVE, 1, (7, 5), (7, 6)) ... etc]
        """
        # array of actions which we will return after we have filled it with possible (legal) moves
        actions = []

        # go through the list of applicable BOOM actions and add them to actions[]
        squares = self.state.get_squares(colour)
        for stack in squares:
            actions.append((BOOM, stack))

        # go through the list of applicable MOVE actions
        # for each item from .items() from a stack dictionary, item[0] is the (x, y) coordinates of of the stack and
        # item[1] is the number of pieces in the stack
        for stack in self.state.get_colour(colour).items():
            # iterate through each possible number of pieces to move from our stack at the current occupied_square
            for n_pieces in range(1, stack[1] + 1):
                # possible moving directions
                for move_direction in MOVE_DIRECTIONS:
                    # number of squares to move n_pieces from current stack, 1 <= n_steps <= n_pieces
                    for n_steps in range(1, stack[1] + 1):
                        # check if moving n_steps in move_direction from current stack is a legal move (i.e. not out of
                        # bounds and not landing on an enemy piece)
                        if is_legal_move(self.state.get_squares(opponent(colour)), stack[0], move_direction, n_steps):
                            final_square = calculate_dest_square(stack[0], move_direction, n_steps)
                            actions.append((MOVE, n_pieces, stack[0], final_square))
        return actions

    def get_children(self, colour):
        children = []
        actions = self.get_possible_actions(colour)
        for action in actions:
            children.append(self.apply_action(colour, action))
        return children


def is_legal_move(enemy_stack_locations, moving_stack_location, move_direction, n_steps):
    """ check if moving n_steps in move_direction from current stack is a legal move (i.e. not out of bounds and not
    landing on an enemy piece) """
    dest_square = calculate_dest_square(moving_stack_location, move_direction, n_steps)
    return bool((dest_square[0] in range(0, 8)) and (dest_square[1] in range(0, 8)) and (
            dest_square not in enemy_stack_locations))


def calculate_dest_square(moving_stack_location, move_direction, n_steps):
    return (
        moving_stack_location[0] + n_steps * move_direction[0], moving_stack_location[1] + n_steps * move_direction[1])


def opponent(colour):
    """get the opponent colour. If given 'white' return 'black' and vice versa"""
    if colour == WHITE:
        return BLACK
    if colour == BLACK:
        return WHITE
    return None


def manhattan_dist(state):
    total = 0
    for white in state.white_stacks.items():
        current_total = 0
        for black in state.black_stacks.items():
            current_total += (abs(white[0][0] - black[0][0]) + abs(white[0][1] - black[0][1]))
        total += current_total / len(state.black_stacks)
    return total / len(state.white_stacks)


def move_action(colour, base_node, n_pieces, stack, dest_square):
    """ apply a move action to the given base node by moving n_pieces from stack n_steps in move_direction
            returns new_node resulting from the move """
    # make a new node that is a copy of the base_node. Set last_player to current colour making the move
    new_node = base_node.copy()
    # adjust new_node fields according to how our move will change them:
    # parent node of the new_node is the base_node
    new_node.parent = base_node
    # new_node depth is parent depth + 1
    new_node.depth = base_node.depth + 1
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
    new_node.value = heuristic(colour, new_node.state)

    return new_node


def boom_action(colour, base_node, stack_to_boom):
    # make a new node that is a copy of the base_node
    new_node = base_node.copy()

    # adjust new_node fields according to how the boom change them:
    # parent node of the new_node is the base_node
    new_node.parent = base_node
    # new_node depth is parent depth + 1
    new_node.depth = base_node.depth + 1
    # store the move which got us to new_node
    new_node.move = (BOOM, stack_to_boom)
    new_node.last_colour = colour

    # recursive boom at the new_node.state starting at 'stack', this updates the state
    new_node.state = chain_boom(new_node.state, stack_to_boom)

    # update value and return
    new_node.value = heuristic(colour, new_node.state)

    return new_node


def heuristic(colour, state):
    if (state.total_black() == 0) and (state.total_white() == 0):
        return DRAW_GAME

    if colour == WHITE:
        if state.total_black() == 0:
            # win game
            return WIN_GAME
        if state.total_white() == 0:
            # lost game
            return LOST_GAME
        # else, the heuristic is the number of our pieces on the board - enemy pieces on the board + manhattan
        # distance, **higher** is better
        return state.total_white() - state.total_black()  # - manhattan_dist(state)

    if colour == BLACK:
        if state.total_white() == 0:
            # win game
            return WIN_GAME
        if state.total_black() == 0:
            # lost game
            return LOST_GAME
        # else, the heuristic is the number of our pieces on the board - enemy pieces on the board + manhattan
        # distance, **higher** is better
        return state.total_black() - state.total_white()  #- manhattan_dist(state)

    # else, incorrect colour given return None
    return None


def is_game_over(state):
    return bool(state.total_black() == 0 or state.total_white() == 0)


def chain_boom(state, stack_to_boom, stacks_to_remove=None):
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
    all_stacks = list(state.white_stacks.keys()) + list(state.black_stacks.keys())
    stacks_hit = list(set(all_stacks).intersection(radius))

    # add all the stacks_stacks_hit to the stacks_to_remove set, if they havent been added before, boom them
    for st in stacks_hit:
        if st not in stacks_to_remove:
            stacks_to_remove.add(st)
            chain_boom(state, st, stacks_to_remove)

    # remove stacks_to_remove from state and return state
    for st in stacks_to_remove:
        if st in state.white_stacks:
            state.white_stacks.pop(st)
        if st in state.black_stacks:
            state.black_stacks.pop(st)
    return state


def get_greedy_action(colour, base_node, budget):
    """returns the action associated with the best score achieved after that action is enacted on our current_node.state
    Note: because python uses a min-heap for their priority queue implementation, better scores are lower, the lower the score the better its value"""

    # store actions in a dict {} where the key is the score achieved by that action, break ties randomly
    # get the possible actions from our current position

    # make a copy of the initial node we were given
    # base_node = Node(current_node.state)
    best_actions = []  # initialise the best_actions with a dummy value so our loop doesnt kick up a fuss when we try to access the [0] index for the first time
    best_score = LOST_GAME
    actions = base_node.get_possible_actions(colour)
    for action in actions:
        current_node = base_node.apply_action(colour, action)
        current_node.value -= manhattan_dist(current_node.state)
        if current_node.value > best_score:
            best_actions = [action]  # reset the list to have 1 action as the new best
            best_score = current_node.value
        elif current_node.value == best_score:
            best_actions.append(action)

    # find the best action in those actions and return it. Break draws randomly
    return random.choice(best_actions)


def get_minimax_action(colour, base_node, budget):
    best_actions = []  # initialise the best_actions with a dummy value so our loop doesnt kick up a fuss when we try to access the [0] index for the first time
    best_score = LOST_GAME
    actions = base_node.get_possible_actions(colour)
    for action in actions:
        current_node = base_node.apply_action(colour, action)
        minimax_evlu = minimax(current_node, budget, False, opponent(colour))[colour]
        if minimax_evlu > best_score:
            best_actions = [action]  # reset the list to have 1 action as the new best
            best_score = minimax_evlu
        elif minimax_evlu == best_score:
            best_actions.append(action)

    # find the best action in those actions and return it. Break draws randomly
    return random.choice(best_actions)


# returns {WHITE: white heuristic, BLACK: black heuristic}
# node = 'current node being worked on'
# depth = the amount of depth we have left to explore
# colour = the current players turn
"""
def minimax(node, depth, maximising_player, colour):
    if depth == 0 or is_game_over(node.state):
        # print({WHITE: heuristic(WHITE, node.state), BLACK: heuristic(BLACK, node.state)})
        return {WHITE: heuristic(WHITE, node.state), BLACK: heuristic(BLACK, node.state)}
    if maximising_player:
        best_value = {colour: LOST_GAME, opponent(colour): WIN_GAME}
        actions = node.get_possible_actions(colour)
        for action in actions:
            current_node = node.apply_action(colour, action)
            tmp = minimax(current_node, depth - 1, False, opponent(colour))
            if tmp[colour] > best_value[colour]:
                best_value[colour] = tmp[colour]
                best_value[opponent(colour)] = tmp[opponent(colour)]
        return best_value
    else:
        best_value = {colour: WIN_GAME, opponent(colour): LOST_GAME}
        actions = node.get_possible_actions(colour)
        for action in actions:
            current_node = node.apply_action(colour, action)
            tmp = minimax(current_node, depth - 1, True, opponent(colour))
            if tmp[colour] < best_value[colour]:
                best_value[colour] = tmp[colour]
                best_value[opponent(colour)] = tmp[opponent(colour)]
        return best_value
"""


def get_alphabeta_action(colour, node, budget):
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


def minimax(node, depth, alpha, beta, maximising_player, colour):
    if depth == 0:
        return heuristic(node.last_colour, node.state)
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

"""def sort_children(colour, children):
    for i in range(1, len(children)):
        key = heuristic(colour, children[i].state)
        j = i - 1
        while j >= 0 and not isinstance(children[j], int) and key < heuristic(colour, children[j].state):
            children[j+1] = children[j]
            j -= 1
        children[j+1] = key"""