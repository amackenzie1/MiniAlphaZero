import numpy as np 
import time 

row = [0, 0, 0, 0, 0, 0, 0]
blank = [row]*6
blank = np.array(blank)

def display(board):
    print("")
    for row in board:
        print("|", end="")
        for i in row:
            if i == -1:
                print(f"{i}|", end="")
            else:
                print(f" {i}|", end="")
        print("")
        print("----------------------")

def human_display(board):
    print("")
    for row in board:
        print("|", end="")
        for i in row:
            if i == -1:
                print(f" X |", end="")
            elif i == 1:
                print(f" O |", end="")
            else:
                print(f"   |", end="")
        print("")
        print("-----------------------------")

check_rows = np.array([
    [1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1],
]).T

check_cols = np.array([
    [1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1]
])

masks = []
for i in range(4):
    for j in range(3):
        mask1 = blank.copy()
        mask2 = blank.copy()
        for k in range(4):
            mask1[j+k][i+k] = 1
            mask2[5-(j+k)][i+k] = 1
        masks.append(mask1)
        masks.append(mask2)

masks = np.array(masks)

def check(board):
    x = (board@check_rows).flatten()
    y = (check_cols@board).flatten()
    z = (masks * board).sum(axis=(1, 2)).flatten()
    t = np.concatenate([x, y, z])
    if t.max() == 4:
        return 1
    elif t.min() == -4:
        return -1
    else:
        return 0

def stringify(board):
    def to_string(board, who):
        if np.count_nonzero(board) == 0:
            return ""
        for i in range(7):
            for j in range(6):
                if board.T[i][j] == who:
                    board.T[i][j] = 0
                    return to_string(board, -1*who) + str(i+1)
                elif board.T[i][j] != 0:
                    continue 
        return "E"

    x = to_string(board, 1)
    if "E" in x:
        return to_string(board, -1)
    else:
        return x

def move(board, move, who):
    impossible = True
    x = 5
    for i in np.flip(board, axis=0):
        if i[move] == 0:
            impossible = False
            break
        x -= 1
    if impossible:
        return None 
    new = board.copy()
    new[x][move] = who 
    return new

def makeboard(string):
  board = blank.copy()
  who = 1
  for i in string:
    board = move(board, int(i)-1, who)
    who *= -1
  return board 

def process_label(label):
  if label == -1:
    return [1, 0, 0]
  elif label == 0:
    return [0, 1, 0]
  else:
    return [0, 0, 1]

def process_board(board):
  positives = (board + np.abs(board))/2 
  negatives = (np.abs(board) - board)/2 
  positives = np.expand_dims(positives, axis=2)
  negatives = np.expand_dims(negatives, axis=2)
  return np.concatenate([positives, negatives], axis=2)
  
if __name__ == '__main__':
    blank[0][0] = 1
    blank[1][0] = 1
    blank[2][0] = 1
    blank[3][0] = 1

    blank[1][1] = -1
    blank[2][2] = -1
    blank[3][3] = -1
    blank[4][4] = -1
    display(blank)
    t1 = time.time()
    check(blank)
    print(time.time() - t1)