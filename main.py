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

command = input("Usted puede elgir que ejercicio realizar. Los ejercicios disponibles son los siguiente:\n1a - Autoencoder basico\n1b - Denoising Autoencoder\n2 - Generacion de nueva muestra.\nElija el ejercicio que prefiera: ")

if command == "1a":
  print("Cargando informacion...\n")
  font = parse_font(font1)
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
  autoencoder.train_weights(font, font)
  end = time.time()
  print("Red entrenada.\n Tiempo transcurrido:")
  print(end - start)
  print(" segundos")
  #mostrar resultados autoencoder

  # #MOSTRAR ENCODER
  # encoder = get_encoder_from_autoencoder(autoencoder, encoder_layers) #modifque para que reciba hasta donde deberia copiar, sabiendo todas las capas.
  # activations = []
  # for char in font:
  #   activations.append(encoder.forward(char))
  # print(activations)
  # # plt.scatter(activations)
  # # plt.show()
  # #MOSTRAR DECODER
  # decoder = get_decoder_from_autoencoder(autoencoder, decoder_layers, latent_layers)
  # activations = []
  # for char in font:
  #   activations.append(decoder.forward(char))
  # print(activations)
  # # plt.scatter(activations)
  # # plt.show()
  
  print("Ejercicio Finalizado.\n")

elif command == "1b":
  print("Cargando informacion...\n")
  # falta
  print("Informacion cargada exitosamente\n")

  print("Ejercicio Finalizado.\n")

elif command == "2":
  print("Cargando informacion...\n")
  # falta
  print("Informacion cargada exitosamente\n")

  print("Ejercicio Finalizado.\n")