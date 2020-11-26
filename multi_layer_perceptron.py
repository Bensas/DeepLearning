import numpy as np
import matplotlib.pyplot as plt
from activation_functions import tanh, dtanh, sigmoide, dsigmoide

class MLP:

    def __init__(self, layers, data_attributes, beta=0.5, start_lr=0.05, end_lr=0.01, max_epochs=10000, activ_function=sigmoide, activ_function_derivative=dsigmoide, momentum=0.4, adaptive_lr=0.00):

        self.layers = layers
        self.weights = []
        self.layer_outputs = [None] * (len(layers))
        self.layer_activations = [None] * (len(layers))
        self.deltas = [None] * (len(layers))
        self.deltaW = [None] * (len(layers))
        self.prevDeltaW = [None] * (len(layers))
        self.learning_rate = start_lr
        self.learning_rate_delta = (end_lr - start_lr) / max_epochs
        self.adaptive_learning_rate_factor = adaptive_lr
        self.error_history = np.array([])
        self.activ_function = activ_function
        self.activ_function_derivative = activ_function_derivative
        self.beta = beta
        self.max_epochs = max_epochs
        self.error = 0
        self.error_threshold = 0.5
        self.momentum = momentum

        for i in range(len(layers)):
            l_out = layers[i]
            l_in  = data_attributes if i == 0 else layers[i-1]
            self.weights.append(np.random.randn(l_in+1,l_out))

    def train_weights(self, data, expected):
        self.error = self.error_threshold + 1
        for epoch in range(self.max_epochs):
            error = 0
            for sample,result in zip(data,expected):
                row = np.array([sample])
                self.forward(row)        
                self.backpropagate(row, result, epoch)
                error += (np.sum(result - self.layer_activations[len(self.layers) - 1]) ** 2)
            self.error = error / len(data)
            self.error_history = np.append(self.error_history, self.error)
            self.adapt_learning_rate_delta()
            self.learning_rate += self.learning_rate_delta
            print("lr:")
            print(self.learning_rate)
            print("error:")
            print(self.error)
            if self.error < self.error_threshold:
                break

    def forward(self, data):
        for i in range(len(self.layers)):
            layer_input = data if i == 0 else self.layer_activations[i - 1]
            layer_input = np.column_stack((layer_input, np.ones((len(layer_input),1))))
            self.layer_outputs[i] = layer_input.dot(self.weights[i])
            self.layer_activations[i] = self.activ_function(self.layer_outputs[i])
        return self.layer_activations[len(self.layers) - 1]
        
    def backpropagate(self, data, expected, epoch):
        output_layer = len(self.layers) - 1
        error_vector = (expected - self.layer_activations[output_layer])
        for i in range(output_layer, -1, -1):
            curr_output = self.layer_outputs[i]
            if i == output_layer:
                self.deltas[i] = error_vector * (self.activ_function_derivative(self.layer_outputs[i]))
            else:
                dt = (self.deltas[i+1].dot(self.weights[i+1].T))
                self.deltas[i] = self.activ_function_derivative(curr_output) * dt[:,:-1]

        for i in range(len(self.layers)):
            prev_activation = data if i == 0 else self.layer_activations[i - 1]
            prev_activation = np.column_stack((prev_activation, np.ones((len(prev_activation),1))))
            self.deltaW[i] = self.learning_rate * ((prev_activation).T).dot(self.deltas[i])
            if self.prevDeltaW[i] is None:
                self.prevDeltaW[i] = np.zeros(self.deltaW[i].shape)

        for i in range(output_layer, -1, -1):
            self.weights[i] = self.weights[i] + self.deltaW[i] + self.momentum * self.prevDeltaW[i]
            self.prevDeltaW[i] = self.deltaW[i]
    
    def adapt_learning_rate_delta(self):
        min_recent_error = self.get_min_recent_error()
        if min_recent_error > self.error:
            self.learning_rate_delta += self.adaptive_learning_rate_factor
        else:
            self.learning_rate_delta -= self.adaptive_learning_rate_factor * self.learning_rate_delta
    
    def get_min_recent_error(self):
        if len(self.error_history) > 10:
            return min(self.error_history[-10:])
        else:
            return min(self.error_history)
    @staticmethod
    def get_encoder_from_autoencoder(autoencoder, encoder_layers):
        result = MLP(encoder_layers, encoder_layers[0])
        result.beta = autoencoder.beta
        result.weights = autoencoder.weights[0:len(encoder_layers)]
        result.error = autoencoder.error
        return result
    
    @staticmethod
    def get_decoder_from_autoencoder(autoencoder, decoder_layers):
        result = MLP(decoder_layers, decoder_layers[0])
        result.beta = autoencoder.beta
        result.weights = autoencoder.weights[-len(decoder_layers):]
        result.error = autoencoder.error
        return result
