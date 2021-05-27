import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Softmax, ZeroPadding2D, BatchNormalization, Activation
from tensorflow.keras.losses import CategoricalCrossentropy 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from utils import display, check, move, makeboard, process_board, process_label

reg = 1e-4

inputs = keras.Input(shape=(6, 7, 2), name="board")
x = Conv2D(64, 3, padding='same', kernel_regularizer=l2(reg))(inputs)
x = BatchNormalization()(x)
residue = Activation(activation='relu')(x)

for i in range(5):
  x = Conv2D(64, 3, padding="same", kernel_regularizer=l2(reg))(residue)
  x = BatchNormalization()(x)
  x = Activation(activation='relu')(x)
  x = Conv2D(64, 3, padding="same", kernel_regularizer=l2(reg))(x)
  x = BatchNormalization()(x)
  x = keras.layers.add([x, residue])
  residue = Activation(activation='relu')(x)

x1 = Conv2D(1, 1, padding='same', kernel_regularizer=l2(reg))(residue)
x1 = BatchNormalization()(x1)
x1 = Activation(activation='relu')(x1)
x1 = Flatten()(x1)
x1 = Dense(65, activation='relu', kernel_regularizer=l2(reg))(x1)
value = Dense(1, activation='tanh', name='value')(x1)
#value = Softmax(name='value')(Dense(3)(x1))

x2 = Conv2D(2, 1, padding='same', kernel_regularizer=l2(reg))(residue)
x2 = BatchNormalization()(x2)
x2 = Activation(activation='relu')(x2)
x2 = Flatten()(x2)
x2 = Dense(7)(x2)
policy = Softmax(name="policy")(x2)


model2 = keras.Model(inputs, outputs=[value, policy], name="mini_alphazero")
model2.summary()

model2.compile(loss={'policy': CategoricalCrossentropy(),
                      'value': 'mse'}, optimizer="adam")

def get_model():
    return model2 