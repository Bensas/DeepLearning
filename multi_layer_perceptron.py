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
            activation_function,
            activation_function_derivate,
            optimizer):
        self.num_of_inputs = num_of_inputs

        self.optimizer = optimizer
        self.layers = layers
        self.weights = [None] * (len(layers))
        self.bias = [None] * (len(layers))
        self.layer_outputs = [None] * (len(layers))
        self.layer_activations = [None] * (len(layers))
        self.deltas = [None] * (len(layers))
        self.deltaW = [None] * (len(layers))
        self.activation_function = activation_function
        self.activation_derivate = activation_function_derivate
        
        for i in range(len(layers)):
            layer_output_counts = layers[i]
            layer_input_counts  = num_of_inputs if i == 0 else layers[i-1]
            self.weights[i] = np.random.randn(layer_input_counts + 1,layer_output_counts)
            self.bias[i] = np.ones((layer_output_counts,1))

    def train_weights(self, data, expected):
        self.input = data
        self.expected = expected
        self.error = 10000

        print(len(self.weights))
        flattened_weights = self.flatten_weights(self.weights)
        print('Minimizing...')
        try:
            res = minimize(self.cost, flattened_weights, method=self.optimizer)
        except RuntimeError:
            print('Hey mona')
        print('Minimized.')
        print("Final error:" + str(self.error))

    def get_encoder_from_autoencoder(self, autoencoder, encoder_layers):
        result = MLP(encoder_layers, encoder_layers[0], None, None, None)
        result.weights = autoencoder.weights[0:len(encoder_layers)]
        result.error = autoencoder.error
        return result

    def get_decoder_from_autoencoder(self, autoencoder, decoder_layers):
        result = MLP(decoder_layers, decoder_layers[0], None, None, None)
        result.weights = autoencoder.weights[-len(decoder_layers):]
        result.error = autoencoder.error
        return result
    
    def cost(self, data):
        self.weights = self.unflatten_weights(data)
        pred = self.predict_weights(self.weights)
        self.error = np.sum((self.expected - pred) ** 2) / len(data)
        if self.error < 0.09:
            raise RuntimeError("Optimization finished :DDDDDDD")
        print("Current error: " + str(self.error))
        return self.error

    def predict(self, data):
        return self.forward(data)

    def forward(self, data):
        for i in range(len(self.layers)):
            layer_input = data if i == 0 else self.layer_activations[i - 1]
            layer_input = np.column_stack((layer_input, np.ones((len(layer_input),1))))
            self.layer_outputs[i] = layer_input.dot(self.weights[i])
            self.layer_activations[i] = self.activation_function(self.layer_outputs[i])
        return self.layer_activations[len(self.layers) - 1]
    
    def predict_weights(self, weights):
        for i in range(len(weights)):
            layer_input = self.input if i == 0 else self.layer_activations[i - 1]
            layer_input = np.column_stack((layer_input, np.ones((len(layer_input),1))))
            self.layer_outputs[i] = layer_input.dot(weights[i])
            self.layer_activations[i] = self.activation_function(self.layer_outputs[i])
        return self.layer_activations[len(self.layers) - 1]

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
