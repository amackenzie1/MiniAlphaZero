import numpy as np
from utils import process_board

def unzip(games):
    moves = []
    for i in games:
        moves += i 
    boards = []
    policies = []
    results = []
    for i in moves:
        boards.append(i[0])
        policies.append(i[1])
        results.append(i[2])
    return boards, policies, results 

def expand(games):
    boards, policies, results = unzip(games)
    flippedboards = np.flip(boards, axis=-1)
    flippedresults = results.copy()
    flippedpolicies = np.flip(policies, axis=-1)
    boards = np.concatenate([boards, flippedboards])
    results = np.concatenate([results, flippedresults])
    policies = np.concatenate([policies, flippedpolicies])
    boards = np.array([process_board(i) for i in boards])

    return boards, policies, results 