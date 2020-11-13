import numpy as np
from numpy import random
import matplotlib.pyplot as plt

class MLP():
    # constructor
    def __init__(self,all_inputs,labels,w_1,w_2,bias,hidden_input_bias,precision,epocas,learning_rate,n_ocultas,n_entradas,n_salida, activation, dactivation):
        # Variables de inicialización 
        self.all_inputs = np.transpose(all_inputs)
        self.labels = labels
        self.w1 = w_1
        self.w2 = w_2
        self.bias = bias
        self.hidden_input_bias = hidden_input_bias
        self.precision = precision
        self.epocas = epocas
        self.learning_rate = learning_rate
        self.n_entradas = n_entradas
        self.n_ocultas = n_ocultas
        self.n_salida = n_salida
        # Variables de aprendizaje
        self.current_label = 0 # Salida deseada en iteracion actual
        self.error_red = 1 # Error total de la red en una conjunto de iteraciones
        self.Ew = 0 # Error cuadratico medio
        self.prev_error = 0 # Error anterior
        self.current_error = np.zeros(len(labels)) # Errores acumulados en un ciclo de muestras
        self.current_inputs = np.zeros((1,n_entradas))
        self.hidden_layer_inputs = np.zeros((n_ocultas,1)) # Entradas en neuronas ocultas
        self.hidden_layer_outputs = np.zeros((n_ocultas,1)) # Resultado de la activacion en neuronas ocultas
        self.output_layer_input = 0.0 # Entrada en la neurona de salida
        self.y = 0.0 # Resultado de la activación en la neurona de salida
        self.current_epochs = 0
        # Variables de retropropagacion
        self.error_real = 0
        self.output_delta = 0.0 # delta de salida
        self.hidden_output_delta = np.zeros((n_ocultas,1)) # Deltas en neuronas ocultas
        self.activation = activation
        self.dactivation = dactivation
        
    def Operacion(self):
        respuesta = np.zeros((len(self.labels),1))
        for p in range(len(self.labels)):
            self.current_inputs = self.all_inputs[:,p]
            self.Propagar()
            respuesta[p,:] = self.y
        return respuesta.tolist()
    
    def Aprendizaje(self, prueba):
        errores = [] # Almacenar los errores de la red en un ciclo
        while(np.abs(self.error_red) > self.precision):
            self.prev_error = self.Ew
            a = 0
            for i in range(len(self.labels)):
                self.current_inputs = self.all_inputs[:,i] # Senales de entrada por iteracion
                self.current_label = self.labels[i]
                self.Propagar()
                self.Backpropagation()
                self.Propagar()
                if(prueba):
                    # Ej3.1
                    
                    if(a == 0):
                        print("esperado para el [-1;1]: " + str(self.current_label))
                    elif(a == 1):
                        print("esperado para el [1;-1]: " + str(self.current_label))
                    elif(a == 2):
                        print("esperado para el [-1;-1]: " + str(self.current_label))
                    else:
                        print("esperado para el [1;1]: " + str(self.current_label))

                    # Ej3.2

                    # print("esperado para el " + str(a) + ": " + str(self.current_label))
                    
                    print("resultado: " + str(self.y))
                self.current_error[i] = (0.5)*((self.current_label - self.y)**2)
                a = a + 1
            # error global de la red
            self.Error()
            errores.append(self.error_red)
            self.current_epochs +=1
            # Si se alcanza un mayor numero de epocas
            if self.current_epochs > self.epocas:
                break
        # Regresar 
        return self.current_epochs,self.w1,self.w2,self.bias,self.hidden_input_bias,errores                
    
    def Propagar(self):
        # Operaciones en la primer capa
        for a in range(self.n_ocultas):
            self.hidden_layer_inputs[a,:] = np.dot(self.w1[a,:], self.current_inputs) + self.hidden_input_bias[a,:]
        
        # Calcular la activacion de la neuronas en la capa oculta
        for o in range(self.n_ocultas):
            self.hidden_layer_outputs[o,:] = self.activation(self.hidden_layer_inputs[o,:])
        
        # Calcular Y potencial de activacion de la neuronas de salida
        self.output_layer_input = (np.dot(self.w2,self.hidden_layer_outputs) + self.bias)
        # Calcular la salida de la neurona de salida
        self.y = self.activation(self.output_layer_input)
    
    def Backpropagation(self):

        self.error_real = (self.current_label - self.y)

        self.output_delta = (self.dactivation(self.output_layer_input) * self.error_real)

        self.w2 = self.w2 + (np.transpose(self.hidden_layer_outputs) * self.learning_rate * self.output_delta)

        self.bias = self.bias + (self.learning_rate * self.output_delta)

        self.hidden_output_delta = self.dactivation(self.hidden_layer_inputs) * np.transpose(self.w2) * self.output_delta

        for j in range(self.n_ocultas):
            self.w1[j,:] = self.w1[j,:] + ((self.hidden_output_delta[j,:]) * self.current_inputs * self.learning_rate)
        
        for g in range(self.n_ocultas):
            self.hidden_input_bias[g,:] = self.hidden_input_bias[g,:] + (self.learning_rate * self.hidden_output_delta[g,:])
        
    def Error(self):
        # Error cuadratico medio
        self.Ew = ((1/len(self.labels)) * (sum(self.current_error)))
        self.error_red = (self.Ew - self.prev_error)
