from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from fonts import font1, font2, font3
from multi_layer_perceptron import MLP
from activation_functions import tanh, dtanh, sigmoide, dsigmoide

def parse_font(font):
    result = np.array([])
    for character in font:
        processed_char = np.array([])
        for column in character:
            # This gets the binary string representation of the number (which starts with the chars 0b) and removed the first two chars, completes with zeroes
            bits = bin(column)[2:].zfill(5)
            for bit in bits:
                processed_char = np.append(processed_char, int(bit))
        result = np.append(result, processed_char, axis=0)
    result = result.reshape(32, 35)
    return result

command = input("Select the desired excercise:")

if command == "1a":
  print("Loading data...")
  font = parse_font(font1)

  print("Data loaded, training network...")
  start = time.time()
  red = MLP([35, 20, 10 , 6, 2, 6, 10, 20, 35], 35, sigmoide, dsigmoide, 'Powell')
  red.train_weights(font, font)
  end = time.time()
  print("Network trained. Elapsed time:")
  print(end - start)

  activations = []
  for char in font:
    activations.append(red.forward(char))
  print(activations)
  # plt.scatter(activations)
  # plt.show()

  print("Exiting.")

elif command == "1b":
  print("Loading data...")

  print("Exiting.")

elif command == "2":
  print("Loading data...")

  print("Exiting.")