from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from fonts import font1, font2, font3
from multicapa import MLP
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
        result = np.append(result, processed_char)
    result = result.reshape(32, 35)
    return result

command = input("Usted puede elgir que ejercicio realizar. Los ejercicios disponibles son los siguiente:\n1a - Autoencoder basico\n1b - Denoising Autoencoder\n2 - Generacion de nueva muestra.\nElija el ejercicio que prefiera:")

if command == "1a":
  print("Cargando informacion...\n")
  font = parse_font(font1)
  print("Informacion cargada exitosamente\n")
  print("Creando del autoenconder:\n")
  layers = [35,25,15,5,2,5,15,25,35]
  n_inputs = 35
  command = input("Seleccione metodo de optimizacion:\n1 - Powell\n2 - BFGS\n3 - Ninguno")
  if command == "1":
    optimizer = "Powell"
  elif command == "2":
    optimizer = "BFGS"
  elif command == "3":
    optimizer = "None"
  autoencoder = MLP(layers, n_inputs, sigmoide, dsigmoide, optimizer)
  #mostrar resultados autoencoder

  #mostrar encoder
  
  #mostrar decoder
  
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