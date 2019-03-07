import numpy as np

def sigmoid(x):
    ## Activation function: f(x) = 1 / (1 + e^(-x)
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        ## Multiply inputs by weight, add bias, apply activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


weights = np.array([0, 1]) ## w_1 = 0, w_2 = 1
bias = 4 ## b = 4
n = Neuron(weights, bias)

x = np.array([2, 3]) ## x_1 = 2, x_2 = 3
print(n.feedforward(x)) # ?

class NeuralNetwork:
    '''
    A neural net with:
    - 2 inputs
    - a hidden layer of 2 neurons
    - an output layer with 1 neuron
    - idential weights and bias for each neuron (w = [0, 1], b=0)
    '''
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0
        ## Hidden layer
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        ## Output layer
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        ## Hidden layer
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        ## Ouput layer
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1

network = NeuralNetwork()
x = 
