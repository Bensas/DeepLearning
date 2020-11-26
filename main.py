from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from noisy_functions import noisy_function_lighter, noisy_function_heavier, hexa_to_binary
from fonts import font1, font2, font3
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
  architecture = [20, 10, 6, 2, 6, 10, 20, 35]
  inputs = 35
  autoencoder = MLP(architecture, inputs)
  for weight in autoencoder.weights:
    print(weight.shape)
  print("Entrenando red...")
  start = time.time()
  autoencoder.train(font, font)
  end = time.time()
  print("Red entrenada.\n Tiempo transcurrido:")
  print(end - start)
  print(" segundos")

  print_nicer(font)
  activ1 = autoencoder.forward(font)
  for arr in activ1:
    index_array = 0
    while(index_array < len(arr)):
      print(arr_step(arr)[index_array:index_array + 5])
      index_array = index_array + 5
    print("\n\n")

  #Latent layer values
  encoder = MLP.get_encoder_from_autoencoder(autoencoder, [35, 20, 10, 6, 2])
  print("Encoder")
  for weight in encoder.weights:
    print(weight.shape)
  activations = encoder.forward(font)

  #Plotting
  plt.title("Espacio latente")
  x, y = activations.T
  plt.scatter(x, y)
  for i, char in enumerate(activations):
    plt.text(char[0], char[1], str(i))
  plt.show() 

  # generate new characters
  decoder = MLP.get_decoder_from_autoencoder(autoencoder, [2, 6, 10, 20, 35])
  print("Decoder")
  for weight in decoder.weights:
    print(weight.shape)
  activations = decoder.forward(activations)
  print(activations)



  # decoder = MLP.get_decoder_from_autoencoder(autoencoder, [2, 6, 10, 20, 35])
  # for i in range(2):
  #   x = random.random()
  #   y = random.random()
  #   print("[" + str(x) + ";" + str(y) + "]\n\n")
  #   activ = decoder.forward([x,y])
  #   index_array = 0
  #   while(index_array < len(activ)):
  #     print(arr_step(activ)[index_array:index_array + 5])
  #     index_array = index_array + 5
  #   print("\n\n")

  # print(activations)
  # plt.scatter(activations)
  # plt.show()
  
  print("Ejercicio Finalizado.\n")

elif command == "1b":
  print("Cargando informacion...\n")
  font = hexa_to_binary(font1)
  noisy_font = noisy_function_heavier(font, 0.05)
  print("Informacion cargada exitosamente\n")
  print("Creando del autoenconder...\n")
  architecture = [35, 20, 10, 6, 2, 6, 10, 20, 35]
  inputs = 35
  autoencoder = MLP(architecture, inputs, start_lr=0.2, end_lr=0.001, adaptive_lr=0.001)
  print("Entrenando red...")
  start = time.time()
  autoencoder.train(noisy_font, font)
  end = time.time()
  print("Red entrenada.\n Tiempo transcurrido:")
  print(end - start)
  print(" segundos\n")
  print("Informacion cargada exitosamente\n")

  test_noisy_font = noisy_function_heavier(font, 0.05)
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