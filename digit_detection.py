import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import imageio

# a = np.zeros([3,2])
# plt.imshow(a, interpolation="nearest")
# plt.show()

class neuralnetwork:
    # this function initializes the number of input hidden and output nodes in the network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.innodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lrate = learningrate
        # these are the matrices for the weights from
        # input to hidden and hidden to the output layers
        self.wih = np.random.rand(self.hnodes, self.innodes) - 0.5
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5

        # alternative to the above for choosing the random weights is:
        # self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # the function normal chooses values from normal dist. and the second parameter is
        # the 1/square root of the number of incoming links
        # it shoulda been inodes i guess but i dont know why it is hnodes???

        #defining the activation function
        self.activation_function = lambda x: sp.expit(x)
    #function to train the network
    def train(self, inputs_list, targets_list):

        # converts a list to 2D array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into the output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # now we calculate the error ie: error = (target - actual)
        output_errors = (targets - final_outputs)
        # error for the hidden layer is the output errors, split by weights recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # we finally update the weights by the formula we found out
        self.who += self.lrate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lrate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass
    #function to query the network
    def query(self, inputs_list):

        # converts input list to 2D array
        inputs = np.array(inputs_list, ndmin=2).T
        # inputs for the hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # output for the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate the signals into the final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        #this gives the final output
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
        pass

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

#creating a neural network
n = neuralnetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# opening the file with the data for training the network
data_file = open("mnist_train_100.csv", 'r')
data_list = data_file.readlines()
data_file.close()

#training the neural network
for records in data_list:
    all_values = records.split(',')
    # scaling the input data
    scaled_values = ((np.asfarray(all_values[1:])/255) * 0.99) + 0.01
    inputs = scaled_values
    #creating the target output values
    onodes = output_nodes
    targets = np.zeros(onodes) + 0.01
    targets[int(all_values[0])] = 0.99

    n.train(inputs, targets)

# #testing the neural network
# test_data_file = open("mnist_test_10.csv", 'r')
# test_data_list = test_data_file.readlines()
# test_data_file.close()
#
# rando = np.random.randint(10)
# all_values = test_data_list[rando].split(',')
# scaled_values1 = ((np.asfarray(all_values[1:])/255)*0.99) + 0.01
# print(n.query(scaled_values1))
#
# #displaying the true value of the data
# scaled_values1 =scaled_values1.reshape((28,28))
# plt.imshow(scaled_values1, interpolation="none", cmap="Greys")
# plt.show()


img_array = imageio.imread("7.png")
img_data = 255.0 - img_array.reshape(784) # we subtract by 255 because the mnist dataset has white=0 and black = 255
img_data = ((np.asfarray(img_data)/255) * 0.99) + 0.01
print(n.query(img_data))

plt.imshow(img_data.reshape((28,28)), interpolation="none", cmap="Greys")
plt.show()