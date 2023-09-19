"""
Tic Tac Toe Player
"""
import copy
import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if terminal(board):
        return False
    else:
        num_of_x = 0
        num_of_O = 0
        for row in board:
            for cell in row:
                if cell == O:
                    num_of_O += 1
                elif cell == X:
                    num_of_x += 1

        return X if num_of_x == num_of_O else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()

    for x in range(3):
        for y in range(3):
            if board[x][y] == EMPTY:
                possible_actions.add((x, y))

    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action not in actions(board):
        raise Exception("Impossible action!")

    new_board = copy.deepcopy(board)

    new_board[action[0]][action[1]] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for sign in [X, O]:
        for i in range(3):
            if (board[0][i] == sign and board[1][i] == sign and board[2][i] == sign) or \
                    (board[i][0] == sign and board[i][1] == sign and board[i][2] == sign):
                return sign

        if (board[0][0] == sign and board[1][1] == sign and board[2][2] == sign) or \
                (board[0][2] == sign and board[1][1] == sign and board[2][0] == sign):
            return sign

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True

    for row in board:
        for cell in row:
            if cell == EMPTY:
                return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    state = winner(board)
    if state == X:
        return 1
    elif state == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    if player(board) == X:
        _, best_action = max_value(board)
        return best_action
    else:
        _, best_action = min_value(board)
        return best_action


def max_value(board):
    if terminal(board):
        return utility(board), None

    v = -math.inf
    best_action = None

    for action in actions(board):
        current_val, _ = min_value(result(board, action))
        if current_val > v:
            v = current_val
            best_action = action

    return v, best_action


def min_value(board):
    if terminal(board):
        return utility(board), None

    v = math.inf
    best_action = None
    
    for action in actions(board):
        current_val, _ = max_value(result(board, action))
        if current_val < v:
            v = current_val
            best_action = action
    return v, best_action
