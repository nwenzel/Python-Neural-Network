### UPDATE 01/28/2013
# Momentum and dynamic learning rates to be launched soon. The code is written, working the bugs out now.



# Code for Basic Python Neural Network #

This repository contains all the Python code for a basic Neural Network that uses back propagation to learn feature weights for each layer in the network architecture. In the "function" version of the code, the code now supports a dynamic learning rate and momentum.

The dynamic learning rate is an accelerator/decelerator that slightly increases the learning rate as the weight updates achieve error and significantly reduces the learning rate and reverses 1/2 the last update when an update increases the error.

The code will allow you to create networks with any number of hidden layers each with any number of units. Input layer and Output layer are determined by training data.

The code works with any number of hidden layers and any number of units in each layer, so go crazy trying out different combinations. BUT... it has not been tested on large data sets, large numbers of hidden layers, or large numbers of units in the hidden layer. So... go crazy at your own risk.

The code requires Numpy (import numpy as np). Matplotlib would be helpful to graph error to check learning process.

## Important Note ##
Neural networks that use back propagation will sometimes fail due to the back prop algorithm landing in a local minimum. From time to time, the algorithm will not return the correct answer and the Cost function will end significantly higher than a correctly trained algorithm.

## Give it a try ##
It's pretty easy to get in and mess with the code. It's all contained in a single file to make it easy to try out.

Well, there are currently 2 files: one implements it as a class, one as a bunch of functions. You only need one or the other: the functions or the class. Pick your poison. Either way, your best bet is to open up iPython to get started.

The basic intuition (hat tip Prof. Ng) of this code is that training data X and Y (numpy arrays) are provided with a user defined list of Theta terms (list of numpy arrays containing weights for each theta from 0 - N where Theta[0] are the weights that transform Layer_0 to Layer_1). The list of Theta terms determine the architecture of the nerual network. The Theta list can be constructed by providing a definition of the hidden layers (a list of integers specifying the number of units for each hidden layer) to initialize_theta( number of input features, number of output classes, list of integers specifying hidden layer units).

initialize_theta(2, 1, [2]) would create a list of Theta terms with dimensions 2X3 and 1X3.
initialize_theta(6, 3, [4, 4]) would create a list of Theta terms with dimensions 4X7, 4X5, and 3X5


### For python_nn_class... ###

At the Terminal:

iPython --pylab


At the iPython prompt:

from python_nn_class import nn_py

nn = nn_py()

target_test, Y_pred, J_list, J_test_list, learning_rates = nn.XOR_test(hidden_unit_length_list = [2], epochs=2500, learning_rate=0.5, momentum_rate=0.1, learning_acceleration=1.05, learning_backup=0.5)


The iPython console should then show the training data (all 4 records), an explanation of the architecture, Theta shapes and values, the cost of the last iteration of the back prop function, and a new set of input data with predicted outputs.

nn.iris_test() and nn.digits_test() work the same as XOR_test.

Additionally, a basic plot of the results of the cost function should also appear.

The class implementation is pretty poor in it's current state. The class method inputs should really be set elsewhere as needed.

### For python_nn_function... ###

At the Terminal:

iPython --pylab

At the iPython prompt:

%run python_nn_function.py

plot(J_list)


The iPython console should then show the training data (all 4 records), an explanation of the architecture, Theta shapes and values, the cost of the last iteration of the back prop function, and a new set of input data with predicted outputs.

The plot() function will show a basic plot of the results of the cost function.

Or, call the test functions directly with:

target_test, Y_pred, J_list, J_test_list, learning_rates, Theta_L = XOR_test(hidden_unit_length_list = [2], epochs=2500, learning_rate=0.5, momentum_rate=0.1, learning_acceleration=1.05, learning_backup=0.5)

iris_test() and digits_test() work the same as XOR_test().

From there, go in and mess with the code.

### Requirements ###
Python 2.7 (may work elsewhere but probably not 3.x)

iPython
Numpy
Matplotlib is recommended but not required

## Author ##

 - Nathan Wenzel, Edge Solutions, Inc. [http://www.edgesolutions.com/](http://www.edgesolutions.com/)

[@nwenzel](http://twitter.com/nwenzel)

## License ##

All source code is copyright (c) 2012, under the Simplified BSD License.  
For more information on FreeBSD see: [http://www.opensource.org/licenses/bsd-license.php](http://www.opensource.org/licenses/bsd-license.php)

All images and materials produced by this code are licensed under the Creative Commons 
Attribution-Share Alike 3.0 United States License: [http://creativecommons.org/licenses/by-sa/3.0/us/](http://creativecommons.org/licenses/by-sa/3.0/us/)
