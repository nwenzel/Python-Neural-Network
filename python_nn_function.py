#!./usr/bin/python
# -*- coding: utf-8 -*-#

import numpy as np

def sigmoid(z):
  """sigmoid is a basic sigmoid function returning values from 0-1"""
  return 1.0 / ( 1.0 + np.exp(-z) )

def sigmoidGradient(z):
  return sigmoid(z) * ( 1 - sigmoid(z) )
  

def initialize_theta(input_unit_count, output_class_count, hidden_unit_length_list=[]):
  """
  initialize_theta creates architecture of neural network
      
  Parameters:
    hidden_unit_length_list - List of hidden layer units
    input_unit_count - integer, number of input units (features)
    output_class_count - integer, number of output classes

  Returns:
    Theta_L - List of theta values with each theta np.array of random values from -1 to 1
  """
  if not hidden_unit_length_list:
    hidden_unit_length_list = [2]

  unit_count_list = [input_unit_count]
  unit_count_list.extend(hidden_unit_length_list)
  unit_count_list.append(output_class_count)

  Theta_L = [ 2 * (np.random.rand( unit_count, unit_count_list[l-1]+1 ) - 0.5) for l, unit_count in enumerate(unit_count_list) if l > 0]
  
  return Theta_L


def print_theta(Theta_L):
  """print_theta() prints Theta_L and architecture info"""

  T = len(Theta_L)

  print
  print 'NN ARCHITECTURE'
  print '%s Layers (%s Hidden)' % ((T + 1), (T-1))
  print '%s Thetas' % T
  print '%s Input Features' % (Theta_L[0].shape[1]-1)
  print '%s Output Classes' % Theta_L[T-1].shape[0]
  print
    
  print 'Units per layer'
  for t, theta in enumerate(Theta_L):
    if t == 0:
      print ' - Input: %s Units' % (theta.shape[1] - 1)
    if t < T-1:
      print ' - Hidden %s: %s Units' % ((t+1), theta.shape[0])
    else:
      print ' - Output: %s Units' % theta.shape[0]
  print

  print 'Theta Shapes'
  for l, theta in enumerate(Theta_L):
    print 'Theta %s: %s' % (l, theta.shape)
  print
  
  print 'Theta Values'
  for l, theta in enumerate(Theta_L):
    print 'Theta %s:' % l
    print theta
  print

def nn_cost(Y, Y_pred):
  """
  nn_cost implements cost function
    
  y is n_observations by n_classes (n_classes = 1 for n_classes <=2)
  Y_pred is predicted y values and must be same shape as y
  """
  if Y.shape != Y_pred.shape:
    if Y.shape[0] != Y_pred.shape:
      raise ValueError,'Wrong number of predictions'
    else:
      raise ValueError,'Wrong number of prediction classes'
    
  n_observations = len(Y)

  # Cost Function
  J = (-1.0/n_observations)*(Y * np.log(Y_pred) + ((1-Y) * np.log(1-Y_pred))).sum()

  return J
  

def nn_predict(X, Theta_L):
  """
  nn_predict calculates activations for all layers given X and thetas in Theta_L

  Parameters
    X is array of input features dimensions n_observations by n_features
    Theta_L

  Returns
    a_N - List of activations for each layer: Input (0), Hidden (1 to T-1), and Output (T)
    a_Z - List of inputs to compute activations for all non-bias units
    a_N[T] - np.Array of predicted Y values with dimensions n_observations by n_classes
  """

  m = len(X)
  T = len(Theta_L)
    
  a_N = []		# List of activations including bias unit for non-output layers
  z_N = []		# List of inputs for non-input layers

  # Input Layer inputs
  a_N.append( X )
  z_N.append( np.zeros((1,1)) ) # add filler Z layer to align lists for a, z, and Theta
  # Loop through each Theta_List theta
  # t is Theta for calculating layer t+1 from layer t
  for t, theta in enumerate(Theta_L):
    # Add bias unit
    if a_N[t].ndim == 1:
      a_N[t].resize(1, a_N[t].shape[0])
    a_N[t] = np.append(np.ones((a_N[t].shape[0],1)), a_N[t], 1)
      
    # Calculate and Append new z and a arrays to z_N and a_N lists
    z_N.append( a_N[t].dot(theta.T) )
    a_N.append( sigmoid(z_N[t+1]) )

  return z_N, a_N, a_N[T]


def back_prop(X_train, Y_train, Theta_L, lmda):
  """
  Parameters
    X_train - np.array of feature inputs with dimensions n_observations by n_features
    Y_train - np.array of class outputs with dimensions n_observations by n_classes
    Theta_L - list (length = Layer_Count - 1) of theta values as np.array with dims Units[l+1] by Units[l]+1
    lmda - number - lambda term for regularization
    
  Returns
    predicted Y values from nn_predict()
    Theta_Gradient_L as list of Theta Gradients
  """

  n_observations = len(X_train)
  T = len(Theta_L)

  # Create variable to hold error caused by each Theta_L[t] in layer a_N[t+1]
  Theta_Gradient_L = [np.zeros_like(theta) for theta in Theta_L]

  # Backprop Error; One list element for each layer
  delta_N = [] # [0 for theta in Theta_L]

  # Forward Pass
  z_N, a_N, Y_pred = nn_predict(X_train, Theta_L)

  # Get Error for Output Layer (linear unit)
  delta = Y_pred - Y_train
  if delta.ndim == 1:
    delta.resize(1, len(delta))
  delta_N.append( delta )

  # Get Error for Hidden Layers working backwards (stop before layer 0; no error in input layer)
  for t in range(T-1,0,-1):
    delta = delta.dot(Theta_L[t][:,1:]) * sigmoidGradient(z_N[t])
    delta_N.append( delta )
  # Reverse the list so that delta_N[t] corresponds to Theta_N[t] numbers delta[t] is delta that Theta[t] causes on a_N[t+1]
  delta_N.reverse()

  # Calculate Gradient from delta and activation
  # t is the Theta from layer t to layer t+1
  for t in range(T):
    Theta_Gradient_L[t] = delta_N[t].T.dot(a_N[t])

  # Create modified copy of the Theta_L for Regularization
  # Coefficient for theta values from bias unit set to 0 so that bias unit is not regularized
  regTheta = [np.zeros_like(theta) for theta in Theta_L]
  for t, theta in enumerate(Theta_L):
    regTheta[t][:,1:] = theta[:,1:]

  # Average Error + regularization penalty  
  for t in range(T):
    Theta_Gradient_L[t] = Theta_Gradient_L[t] * (1.0/n_observations) + (lmda * regTheta[t])
  
  return Y_pred, Theta_Gradient_L


def fit(X_train, Y_train, Theta_L=[], lmda=0, epochs=2):
  """
  fit() calls the training back_prop function for the given number of cycles
  tracks error and error improvement rates
    
  Parameters:
    X_train - np.array of training data with dimension n_observations by n_features
    Y_train - np.array of training classes with dimension n_observations by n_classes
    Theta_L - list of theta terms as np.arrays where each theta has dimensions n_units[l+1] by n_units[l]+1
    epochs -  integer of number of times to update Theta_L
    
  Returns
    List (length = epochs) of result of cost function for each iteration
  """
  
  # If no Theta provided, use a 3 layer architecture with hidden_layer units = 2 or y classes
  if not Theta_L:
    hidden_units = max(2, len(Y[0]), len(X_train[0]))
    Theta_L = initialize_theta([hidden_units])

  J_list = [0] * epochs
  for i in range(epochs):
    # Back prop to get Y_pred and Theta gradient
    Y_pred, Theta_grad = back_prop(X_train, Y_train, Theta_L, lmda)
    # Record cost
    J_list[i] = nn_cost(Y_train, Y_pred)
    # Update Theta using Learning Rate * Theta Gradient
    for t, theta in enumerate(Theta_L):
      # Start with a large learning rate; probably not a good idea, but I'm impatient
      if i < 100:
        learning_rate = 5.0        
      else:
        learning_rate = 1.0
      Theta_L[t] = theta - ( learning_rate * Theta_grad[t] )

  for t, theta in enumerate(Theta_L):
    print 'Theta: %s' % t
    print np.round(theta, 2)

  print 'i:',i,'  - J:',J_list[i]
    
  return Theta_L, J_list


def XOR_test(hidden_unit_length_list = [], epochs=2500):
  """
  XOR_test is a simple test of the nn printing the predicted value to std out
  Trains on a sample XOR data set
  Predicts a single value
  Accepts an option parameter to set architecture of hidden layers
  """
  
  # Set Data for XOR Test  
  X_train = np.zeros((4,2))
  X_train[0,0] = 1.
  X_train[1,1] = 1.
  X_train[2,:] = 1.
  X_train[3,:] = 0.

  Y_train = np.array([1.,1.,0.,0.]).reshape(4,1) 	# Single Class
  #Y_train = np.zeros((4,2))  						# Two classes
  #Y_train[0,0] = 1.
  #Y_train[1,0] = 1.
  #Y_train[2,1] = 1.
  #Y_train[3,1] = 1.

  print 'Training Data: X & Y'
  print X_train
  print Y_train

  # Initialize Theta based on selected architecture
  Theta_L = initialize_theta(X_train.shape[1], Y_train.shape[1], hidden_unit_length_list)
  
  # Fit
  Theta_L, J_list = fit(X_train, Y_train, Theta_L, 1e-5, epochs)
  
  # Predict
  X_new = np.array([[1,0],[0,1],[1,1],[0,0]])
  print 'Given X:'
  print X_new
  z_N, a_N, Y_pred = nn_predict(X_new, Theta_L)
  print 'Predicted Y:'
  print np.round(Y_pred,3)

  return X_train, Y_train, Y_pred, J_list, Theta_L


if __name__ == '__main__':
  """
  Run XOR_test
  Plot Error terms to see learning progress of NN
  """
  X_train, Y_train, Y_pred, J_list, Theta_L =  XOR_test([2], 5000)


