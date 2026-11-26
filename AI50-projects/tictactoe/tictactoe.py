

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():

    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):

    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)

    return X if x_count == o_count else O


def actions(board):

    return {
        (i, j)
        for i in range(3)
        for j in range(3)
        if board[i][j] == EMPTY
    }


def result(board, action):

    if action not in actions(board):
        raise ValueError("Invalid action")

    i, j = action
    new_board = [row.copy() for row in board]
    new_board[i][j] = player(board)
    return new_board


def winner(board):

    lines = []


    lines.extend(board)
    lines.extend([[board[i][j] for i in range(3)] for j in range(3)])


    lines.append([board[i][i] for i in range(3)])
    lines.append([board[i][2 - i] for i in range(3)])

    for line in lines:
        if line[0] is not EMPTY and line[0] == line[1] == line[2]:
            return line[0]

    return None


def terminal(board):

    return winner(board) is not None or not actions(board)


def utility(board):

    game_winner = winner(board)
    if game_winner == X:
        return 1
    if game_winner == O:
        return -1
    return 0


def minimax(board):

    if terminal(board):
        return None

    def value(state, alpha, beta):
        if terminal(state):
            return utility(state)

        if player(state) == X:
            best_value = -math.inf
            for move in actions(state):
                best_value = max(best_value, value(result(state, move), alpha, beta))
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break
            return best_value

        best_value = math.inf
        for move in actions(state):
            best_value = min(best_value, value(result(state, move), alpha, beta))
            beta = min(beta, best_value)
            if alpha >= beta:
                break
        return best_value

    current_player = player(board)
    best_action = None

    if current_player == X:
        best_value = -math.inf
        for move in actions(board):
            move_value = value(result(board, move), best_value, math.inf)
            if move_value > best_value:
                best_value = move_value
                best_action = move
    else:
        best_value = math.inf
        for move in actions(board):
            move_value = value(result(board, move), -math.inf, best_value)
            if move_value < best_value:
                best_value = move_value
                best_action = move

    return best_action
