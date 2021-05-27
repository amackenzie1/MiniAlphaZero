import numpy as np
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Softmax, ZeroPadding2D, BatchNormalization, Activation
from tensorflow.keras.losses import CategoricalCrossentropy 
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam
from utils import display, check, move, makeboard, process_board, process_label, stringify
from model_definition import get_model 
import time 
import pickle 
from expand_boards import expand
import random 
from uuid import uuid1
from multiprocessing import Pool 
import sys 

temp = 0.4
model = get_model()

def get_val(board):
    value = np.squeeze(model(np.array([process_board(board)])).numpy())
    return value @ np.array([-1, 0, 1]).T

def evaluate(board):
    isended = check(board)
    if isended != 0:
        return isended, Softmax()(np.array([0]*7, dtype='float32'))

    val, probs = model(np.array([process_board(board)]))
    return val.numpy()[0][0], np.squeeze(probs)


class Node:
    def __init__(self, board, prob, parent=None, root=False, turn=1, done=False):
        self.board = board
        self.Q = 0
        self.P = prob
        self.N = 0
        self.W = 0 
        self.children = {}
        self.parent = parent 
        self.root = root 
        self.turn = turn 
        self.done = False 


c_puct = 3

class MonteCarloSearchTree:

    def fill(self, node):
        val, probs = evaluate(node.board)
        probs = 0.75 * probs + 0.25 * np.random.dirichlet([1]*7)
        node.W = val
        node.Q = val
        node.N = 1
        node.done = abs(val) == 1
        for i in range(7):
            newboard = node.board.copy()
            newboard = move(newboard, i, 1)
            if newboard is not None:
                node.children[i] = Node(-1*newboard, probs[i], parent=node, turn=-1*node.turn, done=node.done)

        if len(node.children.keys()) == 0:
            node.done = True
            node.W = check(node.board)
            node.Q = node.W
        
    def __init__(self, tau=1):
        self.tau = tau
        self.root = Node(makeboard(""), prob=1, root=True)
        self.fill(self.root)
        self.policy = None
    
        
    def get_move(self):
        distribution = [0 if i not in self.root.children.keys() else self.root.children[i].N for i in range(7)]
        for i in range(7):
            distribution[i] = distribution[i]**(1/self.tau)
        normalization = sum(distribution)
        for i in range(7):
            distribution[i] = distribution[i]/normalization
        self.policy = [round(i, 5) for i in distribution] 
        move = np.random.choice(a=7, p=distribution)
        return move
    
    def search_once(self, node):
        U = {}
        total_visits = 0

        for i in node.children.values():
            total_visits += i.N
        sqrt_total_visits = np.sqrt(total_visits + 1)

        for i in node.children.keys():
            child = node.children[i]
            child_u = c_puct * child.P * sqrt_total_visits / (1 + child.N)
            child_puct = child_u - child.Q 
            U[i] = child_puct 
        
        move = max(U, key=lambda key: U[key])

        if node.children[move].done:
            node.N += 1
            node.W = node.Q * node.N
            self.backup(node, node.Q)

        elif node.children[move].N == 0:
            self.fill(node.children[move])
            self.backup(node.children[move], node.children[move].W)

        else:
            self.search_once(node.children[move])

    def backup(self, node, value):
        if not node.root:
            node.parent.W -= value 
            node.parent.N += 1
            node.parent.Q = node.parent.W/node.parent.N 
            self.backup(node.parent, value*-1)
    
    def search(self, playouts=100):
        for i in range(playouts):
            self.search_once(self.root)

    def advance_root(self, move):
        self.root = self.root.children[move]
        if self.root.N == 0:
            self.fill(self.root)
        if check(self.root.board) != 0:
            return check(self.root.board)
        elif len(self.root.children.keys()) == 0:
            return check(self.root.board)

        return 2

def play_game(tau, depth):
    def get_sequence(length, result):
        x = []
        for i in range(length):
            x.append(result)
            result *= -1
        return x[::-1]

    boards = []
    policies = []

    mcts = MonteCarloSearchTree(tau)
    mcts.search(depth)
    boards.append(mcts.root.board)
    model_move = mcts.get_move()
    policies.append(mcts.policy)
    x = mcts.advance_root(model_move)
    while x == 2:
        mcts.search(depth)
        boards.append(mcts.root.board)
        policies.append(mcts.policy)
        model_move = mcts.get_move() 
        x = mcts.advance_root(model_move)

    boards.append(mcts.root.board)
    policies.append(mcts.policy)
    return list(zip(boards, policies, get_sequence(len(boards), x)))


obvious = []
obvious.append(makeboard("554433"))
obvious.append(makeboard("554466"))
obvious.append(makeboard("545354"))
obvious.append(makeboard("121416"))


def play_vs_random():
    mcts = MonteCarloSearchTree(0.01)
    mcts.search(400)
    model_move = mcts.get_move()
    x = mcts.advance_root(model_move)
    random_move = random.choice(list(mcts.root.children.keys()))
    mcts.advance_root(random_move)

    while x == 2:
        mcts.search(400)
        model_move = mcts.get_move() 
        x = mcts.advance_root(model_move)
        if x != 2:
            return 1

        random_move = random.choice(list(mcts.root.children.keys()))
        x = mcts.advance_root(random_move)

    return -1 

global num_processes
num_processes = 2 
episode_length = 6

def iterate(episode_length):
    games = []

    for i in range(episode_length//num_processes):
        print(f"Process {os.getpid()}, game {i+1}")
        games.append(play_game(1, 400))

    gamefile = str(uuid1())

    pickle.dump(games, open(f"games/{os.uname()[1]}:{gamefile}.p", "wb"))
    
def get_data():
    games = []
    for i in os.listdir("games"):
        games += pickle.load( open( f"games/{i}", "rb" ) )
    for i in os.listdir("games"):
        os.remove(f"games/{i}")
    return games 


def benchmark(length):
    score = 0
    for i in range(length):
        print(f"Benchmark game {i+1}")
        score += play_vs_random()
    
    wins = (score + length)/2 
    return wins/length 

def getversion():
    return int(open("info.txt").readlines()[0].split()[1])


def work(episode_length, boss):
    version = getversion()

    model.load_weights("baby_alphazero/v1")
    iterate(episode_length)
    
    while getversion() == version and not boss:
        print(f"Sleeping, I, {os.getpid()}, am not boss.")
        time.sleep(3)    

def process(episode_length):
    with tf.device("CPU:0"):
        while True:
            work(episode_length, False)


if "baby_alphazero" not in os.listdir():
    model.build(input_shape=(6, 7, 2))
    model.save_weights("baby_alphazero/v1")

episode_length = int(sys.argv[2])
num_processes = int(sys.argv[1])
process(episode_length)