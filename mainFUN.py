import numpy as np


class NeuralNetwork:
    def __init__(self, input_nodes, output_nodes, num_hidden_layer, hidden_layer_sizes, learning_rate,
                 activation_function, use_bias=True):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.num_hidden_layers = int(num_hidden_layer)
        self.lr = learning_rate
        self.use_bias = use_bias
        if activation_function == 'sigmoid':
            self.activation_function = lambda x: 1 / (1 + np.exp(-x))
            self.actv = 'sigmoid'
        elif activation_function == 'tanh':
            self.activation_function = lambda x: np.tanh(x)
            self.actv = 'tanh'
        # Initialize weights for input layer to first hidden layer
        self.weights = [np.random.normal(0.0, pow(hidden_layer_sizes[0], -0.5), (hidden_layer_sizes[0], input_nodes))]

        # Initialize biases for first hidden layer
        # self.biases = [np.zeros((hidden_layer_sizes[0], 1)) for _ in range(self.num_hidden_layers)] if use_bias else [
        #     np.zeros((0, 0)) for _ in range(self.num_hidden_layers)]

        # Initialize weights for connections between hidden layers
        # self.weights += [
        # np.random.normal(0.0, pow(hidden_layer_sizes[i], -0.5), (hidden_layer_sizes[i], hidden_layer_sizes[i - 1]))
        #     for i in range(1, self.num_hidden_layers)]
        self.weights += [np.random.normal(0.0, pow(nodes, -0.5), (nodes, inputs)) for nodes, inputs in
                         zip(hidden_layer_sizes[1:], hidden_layer_sizes[:-1])]

        # Initialize weights for last hidden layer to output layer
        self.weights.append(np.random.normal(0.0, pow(output_nodes, -0.5), (output_nodes, hidden_layer_sizes[-1])))
        # initialize biases for hidden layers nodes
        self.biases = [np.ones((nodes, 1)) for nodes in hidden_layer_sizes] if use_bias else [np.zeros((nodes, 1)) for
                                                                                              nodes in
                                                                                              hidden_layer_sizes]
        # self.biases = [] if not use_bias else [np.ones((nodes, 1)) for nodes in hidden_layer_sizes]
        # Initialize biases for output layer
        # self.biases.append(np.zeros((output_nodes, 1)) if use_bias else np.zeros((0, 0)))
        self.biases.append(np.ones((output_nodes, 1)) if use_bias else np.zeros((output_nodes, 1)))

    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T

        # forward pass

        layers_outputs = [inputs]

        for i in range(self.num_hidden_layers + 1):
            layer_inputs = np.dot(self.weights[i], layers_outputs[i])
            if self.use_bias:
                layer_inputs += self.biases[i]
            layers_outputs.append(self.activation_function(layer_inputs))
        layers_outputs = layers_outputs[1:]

        # Backward_pass

        output_errors = targets - layers_outputs[-1]

        if self.actv == 'sigmoid':
            output_deltas = output_errors * layers_outputs[-1] * (1 - layers_outputs[-1])
        else:
            output_deltas = output_errors * (1 - layers_outputs[-1] ** 2)
        # output_deltas = output_errors * layers_outputs[-1] * (1 - layers_outputs[-1])
        hiddens_delta = [output_deltas]
        j = 0

        for i in range(self.num_hidden_layers, 0, -1):
            errors = np.dot(self.weights[i].T, hiddens_delta[j])

            if self.actv == 'sigmoid':
                hidden_delta = errors * layers_outputs[i - 1] * (1 - layers_outputs[i - 1])
            else:
                hidden_delta = errors * (1 - layers_outputs[i - 1] ** 2)
            # hidden_delta = errors * layers_outputs[i - 1] * (1 - layers_outputs[i - 1])
            hiddens_delta.append(hidden_delta)
            j += 1
        hiddens_delta = hiddens_delta[::-1]

        # update the weights between the input layer and the first hidden layer
        self.weights[0] = self.weights[0] + self.lr * hiddens_delta[0] * inputs.T

        # update weights between the hidden layers
        for i in range(1, len(self.weights)):
            # print(i)
            inpt = layers_outputs[i - 1]
            self.weights[i] = self.weights[i] + self.lr * hiddens_delta[i] * inpt.T
        return layers_outputs

    def transform(self, test_input):
        test_input = np.array(test_input, ndmin=2).T
        layers_outputs = [test_input]
        for i in range(self.num_hidden_layers + 1):

            layer_inputs = np.dot(self.weights[i], layers_outputs[i])
            if self.use_bias:
                layer_inputs += self.biases[i]

            layers_outputs.append(self.activation_function(layer_inputs))

        layers_outputs = layers_outputs[1:]
        return layers_outputs


# n = NeuralNetwork(1, 5, 3, [3, 3, 2], 0.01, 'sigmoid', False)
# ans = n.train([0.1], [0.22, 0.11, 0.2, 0.25, 0.36])
# ans2 = n.transform([0.2])
# print(f"len(n.weights) {len(n.weights)}")
# print(ans2[3])
# print("---------------")
# print(ans[3])
