import numpy as np
import os 
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

model = get_model()
model.load_weights("baby_alphazero/v1")

def get_data():
    games = []
    for i in os.listdir("games"):
        games += pickle.load( open( f"games/{i}", "rb" ) )
    for i in os.listdir("games"):
        os.remove(f"games/{i}")
    return games 


def getversion():
    return int(open("info.txt").readlines()[0].split()[1])

global num_processes
num_processes = 2
episode_length = 6

def train():
    version = getversion()

    model.load_weights("baby_alphazero/v1")
    
    
    while len(os.listdir("games")) < num_processes:
        print(f"Sleeping, I, {os.getpid()} am boss but there aren't enough processes done yet.")
        time.sleep(5)
         
    print(f"Process {os.getpid()} starting training.")
    open("info.txt", "w").write(f"Version: {version+1}")

    window_size = min(20, max(min(version, 4), version//2))*episode_length 
    print(f"Window size: {window_size}")
    if "training_data.p" in os.listdir():
        training_data = pickle.load( open( f"training_data.p", "rb" ) )
    else:
        training_data = [] 
    
    training_data += get_data()
    training_data = training_data[-1*window_size:]
    pickle.dump(training_data, open(f"training_data.p", "wb"))
    random.shuffle(training_data)

    print(f"Length of training data: {len(training_data)}")

    boards, policies, results = expand(training_data)

    print(f"Shape of boards: {boards.shape}") 
    model.fit(boards, {'policy': policies, 'value': results}, epochs=2, batch_size=32)
    model.save_weights("baby_alphazero/v1")

    
num_processes = int(sys.argv[1])
episode_length = int(sys.argv[2])
while True:
    train()