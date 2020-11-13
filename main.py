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

command = input("Select the desired excercise:")

if command == "1a":
  print("Loading data...")
  font = parse_font(font1)
  # Parametros de la red
  fac_ap = 0.2
  precision = 0.00000001
  epocas = 10000 #
  epochs = 0
  # Arquitectura de la red
  n_entradas = font.shape[1] # numero de entradas
  # cap_ocultas = 1 # Una capa oculta
  n_ocultas = 6 # Neuronas en la capa oculta
  n_salida = font.shape[1] # Neuronas en la capa de salida
  # Valor de umbral o bia
  us = 1.0 # umbral en neurona de salida
  uoc = np.ones((n_ocultas,1),float) # umbral en las neuronas ocultas
  # Matriz de pesos sinapticos
  random.seed(0) # 
  w_1 = random.rand(n_ocultas,n_entradas)
  w_2 = random.rand(n_salida,n_ocultas)
  # Funcion de activacion y su derivada.
  funcion = sigmoide
  derivada_funcion = dsigmoide
  #Inicializar la red PMC
  print(w_1)
  print(w_2)
  print("despues ... \n")
  red = MLP(font,font,w_1,w_2,us,uoc,precision,epocas,fac_ap,n_ocultas,n_entradas,n_salida, funcion, derivada_funcion)
  epochs,w1_a,w2_a,us_a,uoc_a,E = red.Aprendizaje(True)
  print(w1_a)
  print(w2_a)
  print("error final: ")
  print(red.error_red)

  print("Exiting.")

elif command == "1b":
  print("Loading data...")

  print("Exiting.")

elif command == "2":
  print("Loading data...")

  print("Exiting.")