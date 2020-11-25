import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime 

class MLP:

    def __init__(
            self,
            layers, 
            num_of_inputs,
            num_of_outputs,
            activation_function,
            activation_function_derivate,
            optimizer='Powell'):
        self.num_of_inputs = num_of_inputs

        self.optimizer = optimizer
        self.layers = layers
        self.weights = [None] * (len(layers)+1)
        self.bias = [None] * (len(layers)+1)
        self.layer_outputs = [None] * (len(layers) + 1)
        self.layer_activations = [None] * (len(layers) + 1)
        self.deltas = [None] * (len(layers) + 1)
        self.deltaW = [None] * (len(layers) + 1)
        self.activation_function = activation_function
        self.activation_derivate = activation_function_derivate
        self.cost_counter = 0
        
        for i in range(len(layers)+1):
            layer_output_counts = num_of_outputs if i == len(layers) else layers[i]
            layer_input_counts  = num_of_inputs if i == 0 else layers[i-1]
            self.weights[i] = np.random.randn(layer_input_counts,layer_output_counts)
            self.bias[i] = np.ones((layer_output_counts,1))
        # print(self.weights)
        for weight in self.weights:
            print(weight.shape)
        # print(self.ds)

    def train_weights(self, data, expected):
        self.input = data
        print('monana')
        print(len(self.input[0]))
        self.expected = expected
        self.error = 10000

        print(len(self.weights))
        flattened_weights = self.flatten_weights(self.weights)
        print('Training...')
        try:
            res = minimize(self.cost, flattened_weights, method=self.optimizer)
        except RuntimeError:
            print('Training finished')
        print("Final error:" + str(self.error))

    @staticmethod
    def get_encoder_from_autoencoder(autoencoder, encoder_layers):
        result = MLP(encoder_layers, encoder_layers[0], encoder_layers[-1], autoencoder.activation_function, autoencoder.activation_derivate)
        result.weights = autoencoder.weights[0:len(encoder_layers)]
        result.error = autoencoder.error
        return result
    
    @staticmethod
    def get_decoder_from_autoencoder(autoencoder, decoder_layers):
        result = MLP(decoder_layers, decoder_layers[0], decoder_layers[-1], autoencoder.activation_function, autoencoder.activation_derivate)
        result.weights = autoencoder.weights[-len(decoder_layers):]
        for weight in result.weights:
            print(len(weight))
        print(result.weights[0])
        print(len(result.weights[0]))
        result.error = autoencoder.error
        return result
    
    def cost(self, new_weights):
        self.cost_counter += 1
        self.weights = self.unflatten_weights(new_weights)
        # print(len(new_weights))
        pred = self.forward(self.input)
        # if self.cost_counter % 10 == 0:
            # print('Pred:')
            # print(pred)
            # print('Expected:')
            # print(self.expected)
            # print("Current error: " + str(self.error))

        self.error = np.sum((self.expected - pred) ** 2)
        if self.error < 0.009:
            raise RuntimeError("Optimization finished :DDDDDDD")
        # print("Current error: " + str(self.error))
        return self.error

    def predict(self, data):
        return self.forward(data)

    def forward(self, data, printTrue=False):
        # print("Heeeey mona")
        print(self.weights[0])
        for i in range(len(self.weights)):
            layer_input = data if i == 0 else self.layer_activations[i - 1]
            if (printTrue):
                print("LI")
                print(layer_input)
            # print(layer_input.shape)
                print("Weight")
                # print(self.weights[i])
            self.layer_outputs[i] = layer_input.dot(self.weights[i])
            self.layer_activations[i] = self.activation_function(self.layer_outputs[i])
            if printTrue:           
                print("LA")
                print(self.layer_activations[i])
            # print(self.layer_outputs[i].shape)
        return self.layer_activations[len(self.weights) - 1]

    def flatten_weights(self, unflattened_weights):
        flattened_weights = unflattened_weights[0].flatten()
        for i in range(len(unflattened_weights)):
            if i > 0:
                flattened_weights = np.concatenate((flattened_weights, unflattened_weights[i].flatten()))
        return flattened_weights

    def unflatten_weights(self, flattened_weights):
        unflattened_weights = [None] * (len(self.weights))
        index = 0
        for i in range(len(self.weights)):
            layer_shape = self.weights[i].shape
            layer_items = layer_shape[0] * layer_shape[1]
            unflattened_weights[i] = flattened_weights[index:index+layer_items].reshape(layer_shape)
            index += layer_items
        return unflattened_weights
