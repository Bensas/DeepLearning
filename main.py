import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fonts import font1, font2, font3

def parse_font(font):
    result = np.array([])
    for character in font:
        processed_char = np.array([])
        for column in character:
            # This gets the binary string representation of the number (which starts with the chars 0b) and removed the first two chars, completes with zeroes
            bits = bin(column)[2:].zfill(5)
            for bit in bits:
                processed_char = np.append(processed_char, int(bit))
        result = np.append(result, processed_char)
    result = result.reshape(32, 35)
    return result

command = input("Select the desired excercise:")

if command == "1a":
  print("Loading data...")
  font = parse_font(font1)

  print("Exiting.")

elif command == "1b":
  print("Loading data...")

  print("Exiting.")

elif command == "2":
  print("Loading data...")

  print("Exiting.")