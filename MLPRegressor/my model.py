                ###################### SEBAXU #########################
                # This is my first project and I am proud of it ,     #
                # I feel hungry now I may free up later and edit it   # 
                # to become a ready-made library, but I'm tired becaus#
                # of the long week I spent with this matter           #
                # I feel really tired Let the ducks spread everywhere #
                ###################### SEBAXU #########################

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, inputs, hidden_layers, hidden_neurons, outputs, learning_rate=0.1):
        self.inputs = inputs
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.weights = []
        self.weights.append(np.random.rand(self.inputs, self.hidden_neurons[0]))
        for i in range(1, self.hidden_layers):
            self.weights.append(np.random.rand(self.hidden_neurons[i-1], self.hidden_neurons[i]))
        self.weights.append(np.random.rand(self.hidden_neurons[-1], self.outputs))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, inputs):
        layer_output = [inputs]
        for i in range(self.hidden_layers):
            layer_output.append(self.sigmoid(np.dot(layer_output[i], self.weights[i])))
        layer_output.append(np.dot(layer_output[-1], self.weights[-1]))
        return layer_output

    def train(self, inputs, labels, epochs):
        for i in range(epochs):
            layer_output = self.forward_pass(inputs)
            layer_error = [labels - layer_output[-1]]
            for j in range(self.hidden_layers, 0, -1):
                layer_error.insert(0, layer_error[0].dot(self.weights[j].T) * self.sigmoid_derivative(layer_output[j]))
            for j in range(self.hidden_layers):
                self.weights[j] += self.learning_rate * np.dot(layer_output[j].T, layer_error[j])
            self.weights[-1] += self.learning_rate * np.dot(layer_output[-2].T, layer_error[-1])

    def predict(self, inputs):
        layer_output = self.forward_pass(inputs)
        return layer_output[-1]

x = np.array([[-10,  -9  ,-8 , -7 , -6,  -5  ,-4  ,-3 , -2 , -1  , 0  , 1 ,  2,   3  , 4 ,  5  , 6 ,  7,   8,  9]])


y = np.array([[14. , 15.8, 17.6 ,19.4, 21.2, 23. , 24.8 ,26.6, 28.4 ,30.2, 32.,  33.8, 35.6, 37.4, 39.2, 41. , 42.8 ,44.6 ,46.4 ,48.2]])

nn = NeuralNetwork(inputs=20, hidden_layers=3, hidden_neurons=[4,4,1], outputs=20)
nn.train(x, y, 100)
pred = nn.predict(x)
print(pred)
fig, ax = plt.subplots()
ax.plot(y.T, '-b', label='Real Temperatures')
ax.plot(pred.T, '--r', label='Predicted Temperatures')
ax.legend()
plt.show()

