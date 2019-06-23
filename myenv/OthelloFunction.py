import copy
import numpy as np

NO_STONE = -1
BLACK = 0
WHITE = 1
BOARD_SIZE = 8

def check_finished(board):
    white_action = get_possible_actions(board,WHITE)
    if len(white_action) != 0:
        return False

    black_action = get_possible_actions(board,BLACK)
    if len(black_action) != 0:
        return False

    return True

def do_action(board,hand,action):
    new_board = copy.deepcopy(board)
    point = action_to_point(action)

    # pass
    if action == 64:
        return new_board

    new_board[point[0]][point[1]] = hand
    new_board = reverse(new_board,point[1],point[0],hand)

    return new_board


def action_to_point(action):
    point = [0,0]
    point[0] = int(action/8)
    point[1] = action%8

    return point

def point_to_action(point):
    action = 8*point[0] + point[1]
    return action

def get_onehot(board,hand):
    board_onehot = np.zeros((8,8,2))
    for y in range(board.shape[0]):
        for x in range(board.shape[1]):
            if board[y][x] == hand:
                board_onehot[y][x][0] = 1
            elif board[y][x] == 1-hand:
                board_onehot[y][x][1] = 1
    return board_onehot

def reverse(board,x, y, hand):
    reverse_list = can_rev(board,x, y, hand)
    new_board = update(board,hand,reverse_list)
    return new_board


def can_rev(board,x, y, hand):
    "get reverse point"
    PREV = -1
    NEXT = 1
    DIRECTION = [PREV, 0, NEXT]
    flippable = []

    for dx in DIRECTION:
        for dy in DIRECTION:
            if dx == 0 and dy == 0:
                continue

            tmp = []
            depth = 0
            while(True):
                depth += 1

                rx = x + (dx * depth)
                ry = y + (dy * depth)

                if 0 <= rx < BOARD_SIZE and 0 <= ry < BOARD_SIZE:
                    request = board[ry][rx]

                    if request == -1:
                        break

                    if request == hand:
                        if tmp != []:
                            flippable.extend(tmp)
                    else:

                        tmp.append((rx, ry))
                else:
                    break
    return flippable

def update(board,hand,reverse_list):
    new_board = copy.deepcopy(board)
    for p in reverse_list:
        new_board[p[1]][p[0]] = hand
    return new_board

def get_reward(board,hand):
    reward = 0
    white_sum = 0
    black_sum = 0
    for y in board:
        for point in y:
            if point == BLACK:
                black_sum += 1
            elif point == WHITE:
                white_sum += 1

    if hand == BLACK:
        if black_sum > white_sum:
            reward = 100
        else:
            reward = -100

    elif hand == WHITE:
        if white_sum > black_sum:
            reward = 100
        else:
            reward = -100

    return reward


def get_possible_actions(board, my_hand):
    actions=[]
    d = BOARD_SIZE
    PREV = -1
    NEXT = 1
    DIRECTION = [PREV, 0, NEXT]
    opp_hand = 1 - my_hand
    for pos_y in range(d):
        for pos_x in range(d):
            if (board[pos_y, pos_x] != NO_STONE):
                continue
            for dy in DIRECTION:
                if pos_y*8 + pos_x in actions:
                    break
                for dx in DIRECTION:
                    if pos_y*8 + pos_x in actions:
                        break
                    if(dx == 0 and dy == 0):
                        continue
                    depth = 0
                    opp_count = 0
                    while(True):
                        depth += 1
                        rx = pos_x + (dx * depth)
                        ry = pos_y + (dy * depth)

                        if 0 <= rx < d and 0 <= ry < d:
                            request = board[ry][rx]
                            if request == -1:
                                break
                            if request == my_hand:
                                if opp_count > 0:
                                    actions.append(pos_y*8 + pos_x)
                            else:
                                opp_count += 1
                        else:
                            break
    return actions

def onehot_to_board(board_onehot):
    board = np.ones((BOARD_SIZE,BOARD_SIZE)) * -1
    for i in range(2):
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if board_onehot[y][x][i] == 1:
                    board[y][x] = i
    return board
