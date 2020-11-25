from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from noisy_functions import noisy_function_lighter, noisy_function_heavier, hexa_to_binary
from fonts import font1, font2, font3
from multi_layer_perceptron import MLP
from activation_functions import tanh, dtanh, sigmoide, dsigmoide

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
  print("Informacion cargada exitosamente\n")
  # command = input("Seleccione metodo de optimizacion:\n1 - Powell\n2 - BFGS\n3 - Newton\n4 - Gradientes Conjugados\n5 - Ninguno\nSeleccione: ")
  # if command == "1":
  #   optimizer = "Powell"
  # elif command == "2":
  #   optimizer = "BFGS"
  # elif command == "3":
  #   optimizer = "Newton-CG"
  # elif command == "4":
  #   optimizer = "CG"
  print("Creando del autoenconder...\n")
  architecture = [35, 20, 10, 6, 2, 6, 10, 20, 35]
  attributes = len(font[0])
  autoencoder = MLP(architecture, attributes)
  print("Entrenando red...")
  start = time.time()
  autoencoder.train_weights(font, font)
  end = time.time()
  print("Red entrenada.\n Tiempo transcurrido:")
  print(end - start)
  print(" segundos")

  print(font)
  activ1 = autoencoder.forward(font)
  for arr in activ1:
    print(arr_step(arr))

  #Latent layer values
  encoder = MLP.get_encoder_from_autoencoder(autoencoder, [35, 25, 15, 5, 2])
  activations = encoder.forward(font)

  #Plotting
  plt.title("Espacio latente")
  x, y = activations.T
  plt.scatter(x, y)
  for i, char in enumerate(activations):
    plt.text(char[0], char[1], str(i))
  plt.show()

  # generate new characters
  decoder = MLP.get_decoder_from_autoencoder(autoencoder, [2, 5, 15, 25, 35])
  for i in range(2):
    activ = arr_step(decoder.forward(np.random.randn(1, 2))[0])
    print(activ)

  # print(activations)
  # plt.scatter(activations)
  # plt.show()
  
  print("Ejercicio Finalizado.\n")

elif command == "1b":
  print("Cargando informacion...\n")
  font = hexa_to_binary(font1)
  noisy_font = noisy_function_heavier(font, 0.5)
  print("Informacion cargada exitosamente\n")
  print("Creando del autoenconder...\n")
  encoder_layers = [35, 20, 10, 6, 2]
  decoder_layers = [2, 6, 10, 20, 35]
  layers = put_layers_together(encoder_layers, decoder_layers)
  n_inputs = encoder_layers[0]
  command = input("Seleccione metodo de optimizacion:\n1 - Powell\n2 - BFGS\n3 - Newton\n4 - Gradientes Conjugados\n5 - Ninguno\nSeleccione: ")
  if command == "1":
    optimizer = "Powell"
  elif command == "2":
    optimizer = "BFGS"
  elif command == "3":
    optimizer = "Newton-CG"
  elif command == "4":
    optimizer = "CG"
  autoencoder = MLP(layers, n_inputs, sigmoide, dsigmoide, optimizer)
  print("Entrenando red...")
  start = time.time()
  autoencoder.train_weights(noisy_font, font)
  end = time.time()
  print("Red entrenada.\n Tiempo transcurrido:")
  print(end - start)
  print(" segundos\n")
  print("Informacion cargada exitosamente\n")

  print("Ejercicio Finalizado.\n")

elif command == "2":
  data = parse_font(font1)
  architecture = [35, 25, 17, 10, 6, 10, 17,25, 35]
  attributes = len(data[0])
  mlp = MLP(architecture, attributes)
  mlp.train_weights(data, data)

  print(data)
  activ1 = mlp.forward(data)
  for arr in activ1:
    print(arr_step(arr))
  print("Cargando informacion...\n")
  # falta
  print("Informacion cargada exitosamente\n")

  print("Ejercicio Finalizado.\n")