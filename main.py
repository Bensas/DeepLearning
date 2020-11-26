from numpy import random
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from noisy_functions import noisy_function_lighter, noisy_function_heavier, hexa_to_binary
from fonts import font1, font2, font3, cyrillic
from multi_layer_perceptron import MLP
from activation_functions import tanh, dtanh, sigmoide, dsigmoide

def print_nicer(array):
  index_array = 0
  while(index_array < len(array)):
    index_inside = 0
    while(index_inside < len(array[index_array])):
      print(array[index_array][index_inside:index_inside + 5])
      index_inside = index_inside + 5
    index_array = index_array + 1
    print("\n\n")
  return

def put_layers_together(encoder_layers, decoder_layers):
  index = 0
  layers = [0] * (len(encoder_layers) + len(decoder_layers) - 1)
  for number in encoder_layers:
    if index < len(encoder_layers) - 1:
      layers[index] = number
      index = index + 1
  for number in decoder_layers:
    layers[index] = number
    index = index + 1
  return layers

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
  result = result.reshape(len(font), 35)
  return result

def step(num):
  return 0 if num < 0.5 else 1

def arr_step(input_arr):
  result = []
  for i, number in enumerate(input_arr):
    result.append(step(number))
  return np.array(result)

command = input("Usted puede elgir que ejercicio realizar. Los ejercicios disponibles son los siguiente:\n1a - Autoencoder basico\n1b - Denoising Autoencoder\n2 - Generacion de nueva muestra.\nElija el ejercicio que prefiera: ")

if command == "1a":
  
  print("Cargando informacion...\n")
  font = parse_font(font1)
  print(font)
  print("Informacion cargada exitosamente\n")
  print("Creando el autoenconder...\n")
  architecture = [35, 20, 10, 6, 2, 6, 10, 20, 35]
  autoencoder = MLP(architecture)
  print("Entrenando red...")
  start = time.time()
  autoencoder.train(font, font)
  end = time.time()
  print("Red entrenada.\n Tiempo transcurrido:")
  print(end - start)
  print(" segundos")

  #1)a)3)Plot latent layer values
  encoder = MLP.get_encoder_from_autoencoder(autoencoder, [35, 20, 10, 6, 2])
  activations = encoder.forward(font)
  plt.title("Espacio latente")
  x, y = activations.T
  plt.scatter(x, y)
  for i, char in enumerate(activations):
    plt.text(char[0], char[1], str(i))
  plt.show() 

  #1)a)4)Generate new characters
  decoder = MLP.get_decoder_from_autoencoder(autoencoder, [2, 6, 10, 20, 35])
  activations = decoder.forward(activations)
  
  print("Ejercicio Finalizado.\n")

elif command == "1b":
  print("Cargando informacion...\n")
  font = hexa_to_binary(font1)
  print("Informacion cargada exitosamente. Aplicando ruido a la fuente...\n")
  noisy_font = noisy_function_heavier(font, 0.08)
  print("Ruido aplicado. Creando el autoenconder...\n")
  architecture = [35, 20, 10, 6, 2, 6, 10, 20, 35]
  autoencoder = MLP(architecture, start_lr=0.08, end_lr=0.001)
  print("Entrenando red...")
  start = time.time()
  autoencoder.train(noisy_font, font)
  end = time.time()
  print("Red entrenada.\n Tiempo transcurrido:")
  print(end - start)
  print(" segundos\n")

  test_noisy_font = noisy_function_heavier(font, 0.5)

  activ = autoencoder.forward(test_noisy_font)
  print_nicer(test_noisy_font)
  for arr in activ:
    index_array = 0
    while(index_array < len(arr)):
      print(arr_step(arr)[index_array:index_array + 5])
      index_array = index_array + 5
    print("\n\n")

  print("Ejercicio Finalizado.\n")

elif command == "2":
  print("Cargando informacion...\n")
  font = cyrillic
  print("Informacion cargada exitosamente\n")
  print("Creando el autoenconder...\n")
  architecture = [35, 20, 10, 6, 2, 6, 10, 20, 35]
  autoencoder = MLP(architecture)
  print("Entrenando red...")
  start = time.time()
  autoencoder.train(font, font)
  end = time.time()
  print("Red entrenada.\n Tiempo transcurrido:")
  print(end - start)
  print(" segundos")

  # Generate new characters
  decoder = MLP.get_decoder_from_autoencoder(autoencoder, [2, 6, 10, 20, 35])

  x1 = random.random()
  x2 = random.random()
  x3 = random.random()
  y1 = random.random()
  y2 = random.random()
  y3 = random.random()
  new_letters = [[x1, y1],[x2, y2],[x3, y3]]
  activ1 = decoder.forward(new_letters)
  index = 0
  for arr in activ1:
    index_array = 0
    print("Numero generado al azar:\n")
    print(new_letters[index])
    index = index + 1
    while(index_array < len(arr)):
      print(arr_step(arr)[index_array:index_array + 5])
      index_array = index_array + 5
    print("\n\n")

  print("Ejercicio Finalizado.\n")