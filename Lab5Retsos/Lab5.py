import numpy as np

# step 1: load the data
def load_data():
    X_train = np.loadtxt("train_features.dat")  # load train features
    Y_train = np.loadtxt("train_labels.dat").reshape(-1, 1)  # load train labels
    X_test = np.loadtxt("test_features.dat")  # load test features
    Y_test = np.loadtxt("test_labels.dat").reshape(-1, 1)  # load test labels
    return X_train, Y_train, X_test, Y_test

# step 2: define activation functions
def act_sigmoid(x, use_deriv=False):
    if use_deriv:
        return x * (1 - x)  # sigmoid derivative
    return 1 / (1 + np.exp(-x))  # sigmoid function

def act_relu(x, use_deriv=False):
    if use_deriv:
        return np.where(x > 0, 1, 0)  # relu derivative
    return np.maximum(0, x)  # relu function

# step 3: define a single neuron
def neuron(x, w, b, f_act):
    return f_act(np.dot(x, w) + b)  # compute neuron output

# step 4: feed forward computation
def feed_forward(x, W, B, f_act_hid, f_act_out):
    layers_output = []  # store outputs for all layers
    input_layer = x

    for i in range(len(W) - 1):  # process hidden layers
        hidden_output = f_act_hid(np.dot(input_layer, W[i]) + B[i])
        layers_output.append(hidden_output)
        input_layer = hidden_output  # update input for next layer

    # process output layer
    output_layer = f_act_out(np.dot(input_layer, W[-1]) + B[-1])
    layers_output.append(output_layer)

    return layers_output  # list of outputs from each layer

# step 5: back propagation
def back_propagate(Y, yt, W, f_act_hid, f_act_out):
    E = []  # store error for each layer

    # output layer error
    error_output = (yt - Y[-1]) * f_act_out(Y[-1], use_deriv=True)
    E.insert(0, error_output)

    # hidden layer errors
    for i in range(len(W) - 1, 0, -1):
        error_hidden = np.dot(E[0], W[i].T) * f_act_hid(Y[i - 1], use_deriv=True)
        E.insert(0, error_hidden)

    return E  # list of errors for all layers

# step 6: update weights and biases
def update_neurons(x, Y, W, B, E, alpha):
    for i in range(len(W)):  # loop through layers
        layer_input = x if i == 0 else Y[i - 1]  # input to layer
        W[i] += alpha * np.dot(layer_input.T, E[i])  # update weights
        B[i] += alpha * np.sum(E[i], axis=0, keepdims=True)  # update biases

# step 7: loss function
def loss_function(y, yt):
    return np.mean((yt - y) ** 2)  # mse loss

# step 8: initialize weights and biases
def initialize(nh, nn, nxc):
    W = [np.random.randn(nxc if i == 0 else nn, nn) for i in range(nh)] + [np.random.randn(nn, 1)]
    B = [np.random.randn(1, nn) for _ in range(nh)] + [np.random.randn(1, 1)]
    return W, B

# step 9: train the mlp
def mlp(nh, nn, ne, alpha, X_train, Y_train):
    nxc = X_train.shape[1]  # number of input features
    W, B = initialize(nh, nn, nxc)  # initialize weights and biases
    losses = []  # track losses over epochs

    for epoch in range(ne):
        for i in range(X_train.shape[0]):  # loop through each sample
            Y = feed_forward(X_train[i:i+1], W, B, act_relu, act_sigmoid)  # forward pass
            E = back_propagate(Y, Y_train[i:i+1], W, act_relu, act_sigmoid)  # backpropagation
            update_neurons(X_train[i:i+1], Y, W, B, E, alpha)  # update parameters

        # calculate loss after epoch
        output = feed_forward(X_train, W, B, act_relu, act_sigmoid)[-1]
        loss = loss_function(output, Y_train)
        losses.append(loss)

    return W, B, losses  # return final weights, biases, and losses

# step 10: test predictions
def predictions(X_test, Y_test, W_train, B_train):
    output = feed_forward(X_test, W_train, B_train, act_relu, act_sigmoid)[-1]  # test predictions
    predictions = np.round(output)  # round to binary
    accuracy = np.mean(predictions == Y_test)  # calculate accuracy
    return accuracy

# main: run experiments and print summary
if __name__ == "__main__":
    # load data
    X_train, Y_train, X_test, Y_test = load_data()

    results = []  # store results for all configurations

    # base accuracy (untrained model)
    W_base, B_base = initialize(2, 4, X_train.shape[1])  # random initialization
    base_accuracy = predictions(X_test, Y_test, W_base, B_base)
    results.append(("Base Accuracy (Untrained)", base_accuracy))

    # step 5: varying hidden layers
    for nh in [2, 3, 4]:
        accuracy = predictions(X_test, Y_test, *mlp(nh=nh, nn=4, ne=1000, alpha=0.125, X_train=X_train, Y_train=Y_train)[:2])
        results.append((f"{nh} hidden layers, 4 neurons each", accuracy))

    # step 6: varying neurons per layer
    for nn in [4, 8, 16]:
        accuracy = predictions(X_test, Y_test, *mlp(nh=2, nn=nn, ne=1000, alpha=0.125, X_train=X_train, Y_train=Y_train)[:2])
        results.append((f"2 hidden layers, {nn} neurons each", accuracy))

    # step 7: varying learning rates
    for alpha in [0.125, 0.0625, 0.03125]:
        accuracy = predictions(X_test, Y_test, *mlp(nh=2, nn=4, ne=1000, alpha=alpha, X_train=X_train, Y_train=Y_train)[:2])
        results.append((f"2 hidden layers, 4 neurons each, learning rate {alpha}", accuracy))

    # print all results
    print("\nSummary of Results:")
    for config, acc in results:
        print(f"{config}: Accuracy = {acc * 100:.2f}%")
