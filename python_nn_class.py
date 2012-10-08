#!./usr/bin/python
# -*- coding: utf-8 -*-#

import numpy as np
from matplotlib.pyplot import plot

class py_nn():
  """
  py_nn() implements a python neural net with back propagation used to fit the theta parameters.
  Uses Numpy as np
  Uses matplotlib.pyplot.plot

  Initialize:  nn = py_nn()

  nn input and output units determined by training data

  Set nn.hidden_layer_length_list to list of integers to create hidden layer architecture
  nn.hidden_layer_length_list does not include the bias unit for each layer
  nn.hidden_layer_length_list is list containing number of units in each hidden layer
    [4, 4, 2] will create 3 hidden layers of 4 units, 4 units, and 2 units
    Entire architecture will be 5 layers with units [n_features, 4, 4, 2, n_classes]

  nn.fit(X, Y, epochs) where X is training data np.array of features, Y is training data of np.array of output classes , epochs is integer specifying the number of training iterations
  For multi-class prediction, each observation in Y should be implemented as a vector with length = number of classes where each position represents a class with 1 for the True class and 0 for all other classes
  For multi-class prediction, Y will have shape n_observations by n_classes

  nn.nn_predict(X) returns vector of probability of class being true or false
  For multi-class prediction, returns a vector for each observation will return a vector where each position in the vector is the probability of a class
  
  Test a simple XOR problem with nn.XOR_test()
  nn.XOR_test() accepts an optional list of integers to determine the hidden layer architecture
  """

  def __init__(self):

    self.Theta_L = [] 			# List of Theta numpy.arrays
    self.a_N = []				# List of activations including bias unit for non-output layers
    self.z_N = []				# List of inputs for non-input layers
    self.J_list = []			# List of error values
    self.lmda = 1e-5			# Regularization term
    self.hidden_layer_length_list = []
    self.reset_theta() 			# Sets self.hidden_layer_length_list to [2]
    
  def reset_theta(self):
    """self.reset_theta sets theta as a single hidden layer with 2 hidden units"""
    self.hidden_layer_length_list = [2]

  def sigmoid(self, z):
    """sigmoid is a basic sigmoid function returning values from 0-1"""
    return 1.0 / ( 1.0 + np.exp(-z) )

  def sigmoidGradient(self, z):
    return self.sigmoid(z) * ( 1 - self.sigmoid(z) )
  
  def nn_predict(self, X):
    """
    nn_predict calculates activations for all layers, returns prediction for Y

    Parameters
      X is array of input features dimensions n_observations by n_features

    Updates
      self.a_N
      self.a_Z

    Returns
      self.a_N[L] is array of predicted Y values dimensions n_observations by n_classes
    """

    m = len(X)
    T = len(self.Theta_L)
    
    self.a_N = []		# List of activations including bias unit for non-output layers
    self.z_N = []		# List of inputs for non-input layers

    # Input Layer inputs
    self.a_N.append( X )
    self.z_N.append( np.zeros((1,1)) ) # add filler Z layer to align lists by layer
    # Loop through each Theta_List theta
    # t is Theta for calculating layer t+1 from layer t
    for t, theta in enumerate(self.Theta_L):
      # Add bias unit
      if self.a_N[t].ndim == 1:
        self.a_N[t].resize(1, self.a_N[t].shape[0])
      self.a_N[t] = np.append(np.ones((self.a_N[t].shape[0],1)), self.a_N[t], 1)
      
      # Calculate and Append new z and a arrays to z_N and a_N lists
      self.z_N.append( self.a_N[t].dot(theta.T) )
      self.a_N.append( self.sigmoid(self.z_N[t+1]) )

    return self.a_N[T]


  def nn_cost(self, Y, pred_Y):
    """
    nn_cost implements cost function
    
    y is n_observations by n_classes (n_classes = 1 for n_classes <=2)
    pred_y is predicted y values and must be same shape as y
    """
    if Y.shape != pred_Y.shape:
      if Y.shape[0] != pred_Y.shape:
        raise ValueError,'Wrong number of predictions'
      else:
        raise ValueError,'Wrong number of prediction classes'
    
    n_observations = len(Y)

    # Cost Function
    if Y.shape[1] == 1:
      # Y[i] is scalar
      J = (-1.0/n_observations)*(Y.T.dot(np.log(pred_Y)) + (1-Y.T).dot(np.log(1-pred_Y))).sum()
    else:
      # Y[i] is vector
      J = (-1.0/n_observations)*(np.diag(Y.T.dot(np.log(pred_Y))) + np.diag((1-Y.T).dot(np.log(1-pred_Y)))).sum()

    return J
  
  
  def back_prop(self, X, Y):
    """
    X is n_observations, n_features
    Y is n_observations, n_classes
    
    Returns
      predicted Y values
      Theta_Gradient_L
    """


    n_observations = len(X)
    pred_Y = np.zeros_like(Y)
    T = len(self.Theta_L)

    # Create Modified copy of the Theta_L for Regularization
    # Coefficient for bias unit set to 0 so that bias unit is not regularized
    regTheta = [np.zeros_like(theta) for theta in self.Theta_L]
    for t, theta in enumerate(self.Theta_L):
      regTheta[t][:,1:] = theta[:,1:]

    # Create variable to accumulate error caused by each Theta_L term in layer a_N[n+1]
    Theta_Gradient_L = [np.zeros_like(theta) for theta in self.Theta_L]

    for n, x in enumerate(X):
    
      #print 'X[n] VALUE:',n, ' ', x
      
      # Forward Pass
      pred_Y[n] = self.nn_predict(x)
      
      # Backprop Error
      delta_N = []
    
      # Error for Output layer is predicted value - Y training value
      delta = pred_Y[n] - Y[n]
      if delta.ndim == 1:
        delta.resize( 1, len(delta) )
      delta_N.append( delta )
      # Loop backwards through hidden Layers (no error in input layer)
      for t in range(T-1,0,-1):
        delta = delta.dot(self.Theta_L[t][:,1:]) * self.sigmoidGradient(self.z_N[t])
        delta_N.append( delta )
      # Reverse the list so that deltas correspond to theta numbers delta[t] is delta that Theta[t] causes on a_N[t+1]
      delta_N.reverse()
    
      # Accumulate the error terms (no error in input layer)
      # t is the Theta from layer t to layer t+1
      for t in range(T):
        Theta_Gradient_L[t] = Theta_Gradient_L[t] + delta_N[t].T.dot(self.a_N[t])
  
    # Average Error + regularization penalty  
    for t in range(T):
      Theta_Gradient_L[t] = Theta_Gradient_L[t] * (1.0/n_observations) + (self.lmda * regTheta[t])
  
    return pred_Y, Theta_Gradient_L


  def XOR_test(self, hidden_unit_length_list = [], epochs=2500):
    """
      XOR_test is a simple test of the nn printing the predicted value to std out
      Trains on a sample XOR data set
      Predicts a single value
      Accepts an option parameter to set architecture of hidden layers
    """
    
    if not hidden_unit_length_list:
      hidden_unit_length_list = self.hidden_layer_length_list
    
    X = np.zeros((4,2))
    X[0,0] = 1.
    X[1,1] = 1.
    X[2,:] = 1.
    X[3,:] = 0.

    Y = np.array([1.,1.,0.,0.]).reshape(4,1)
    Y = np.zeros((4,2))
    Y[0,0] = 1.
    Y[1,0] = 1.
    Y[2,1] = 1.
    Y[3,1] = 1.

    print 'Training Data: X & Y'
    print np.append(X, Y, 1)
    
    self.initialize_theta(X.shape[1], Y.shape[1], hidden_unit_length_list)
    self.print_theta()
    self.J_list = self.fit(X, Y, epochs)

    X_new = np.array([[1,0],[0,1],[1,1],[0,0]])
    print 'Given X:'
    print X_new
    y_pred = self.nn_predict(X_new)
    print 'Predicted Y:'
    print np.round(y_pred,2)
    plot(self.J_list)


  def initialize_theta(self, input_unit_count, output_class_count, hidden_unit_length_list):
    """
      create_theta creates architecture of neural network
      Defines self.Theta_L
      
      Parameters:
        hidden_unit_length_list - List of hidden layer units
        input_unit_count - integer, number of input units (features)
        output_class_count - integer, number of output classes
    """

    if not hidden_unit_length_list:
      hidden_unit_length_list = self.hidden_layer_length_list
    else:
      self.hidden_layer_length_list = hidden_unit_length_list

    self.Theta_L = []

    unit_count_list = [input_unit_count]
    unit_count_list.extend(hidden_unit_length_list)
    unit_count_list.append(output_class_count)

    self.Theta_L = [ np.random.rand( unit_count, unit_count_list[l-1]+1 ) for l, unit_count in enumerate(unit_count_list) if l > 0]


  def print_theta(self):
    """print_theta(self) prints self.Theta_L and architecture info to std out"""

    self.Theta_L
    T = len(self.Theta_L)

    print
    print 'NN ARCHITECTURE'
    print '%s Layers (%s Hidden)' % ((T + 1), (T-1))
    print '%s Thetas' % T
    print '%s Input Features' % (self.Theta_L[0].shape[1]-1)
    print '%s Output Classes' % self.Theta_L[T-1].shape[0]
    print
    
    print 'Units per layer'
    for t, theta in enumerate(self.Theta_L):
      if t == 0:
        print ' - Input: %s Units' % (theta.shape[1] - 1)
      if t < T-1:
        print ' - Hidden %s: %s Units' % ((t+1), theta.shape[0])
      else:
        print ' - Output: %s Units' % theta.shape[0]
    print
    
    print 'Theta Shapes'
    for l, theta in enumerate(self.Theta_L):
      print 'Theta %s: %s' % (l, theta.shape)
    print
    

  def fit(self, X, Y, epochs):
    """
    main calls the training back_prop function for the given number of cycles
    main tracks error and error improvement rates
    
    parameters
      X, array of training data
      Y, array of training classes
      epochs, integer of number of times to update Theta_L
    
    Updates
      self.Theta_L values
      
    returns
      Final result of Cost Function
    """

    self.J_list = [0] * epochs
    for i in range(epochs):
      # Back prop to get pred_Y and Theta gradient
      pred_Y, Theta_grad = self.back_prop(X, Y)
      # Record cost
      self.J_list[i] = self.nn_cost(Y, pred_Y)
      # Update Theta using Learning Rate * Theta Gradient
      for t, theta in enumerate(self.Theta_L):
        # Start with a large learning rate
        if i < 100:
          self.Theta_L[t] = theta - 5.0 * Theta_grad[t]
        else:
          self.Theta_L[t] = theta - 1.0 * Theta_grad[t]

    for t, theta in enumerate(self.Theta_L):
      print 'Theta: %s' % t
      print np.round(theta, 2)

    print 'i:',i,'  - J:',self.J_list[i]
    
    return self.J_list