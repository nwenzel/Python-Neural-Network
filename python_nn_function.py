#!./usr/bin/python
# -*- coding: utf-8 -*-#

import numpy as np
from sklearn.datasets import load_iris, load_digits

def sigmoid(z):
  """sigmoid is a basic sigmoid function returning values from 0-1"""
  return 1.0 / ( 1.0 + np.exp(-z) )

def sigmoidGradient(z):
  # Not used
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
  
  Returns J - list of cost values
  """
  if Y.shape != Y_pred.shape:
    if Y.shape[0] != Y_pred.shape:
      raise ValueError,'Wrong number of predictions'
    else:
      raise ValueError,'Wrong number of prediction classes'
    
  n_observations = len(Y)
  tiny = 1e-6
  # Cost Function
  J = (-1.0/n_observations)*(Y * np.log(Y_pred + tiny) + ((1-Y) * np.log(1-Y_pred + tiny))).sum()

  return J
  

def nn_predict(X, Theta_L):
  """
  nn_predict calculates activations for all layers given X and thetas in Theta_L

  Parameters
    X is array of input features dimensions n_observations by n_features
    Theta_L

  Returns
    a_N - List of activations for each layer: Input (0), Hidden (1 to T-1), and Output (T)
    a_N[T] - np.Array of predicted Y values with dimensions n_observations by n_classes
  """

  m = len(X)
  T = len(Theta_L)
    
  a_N = []		# List of activations including bias unit for non-output layers

  # Input Layer inputs
  a_N.append( X )
  
  # Loop through each Theta_List theta
  # t is Theta for calculating layer t+1 from layer t
  for t, theta in enumerate(Theta_L):
    # Add bias unit
    if a_N[t].ndim == 1:
      a_N[t].resize(1, a_N[t].shape[0])
    a_N[t] = np.append(np.ones((a_N[t].shape[0],1)), a_N[t], 1)
      
    # Calculate and Append new z and a arrays to z_N and a_N lists
    z = a_N[t].dot(theta.T)
    a_N.append( sigmoid(z) )
  return a_N, a_N[T]


def back_prop(a_N, Y_train, Theta_L, lmda):
  """
  Parameters
    a_N - list of layer outputs with dimensions n_observations by n_units
    Y_train - np.array of class outputs with dimensions n_observations by n_classes
    Theta_L - list (length = Layer_Count - 1) of theta values as np.array with dims Units[l+1] by Units[l]+1
    lmda - number - lambda term for regularization
    
  Returns
    Theta_Gradient_L as list of Theta Gradients
  """
  T = len(Theta_L)
  Y_pred = a_N[T]
  n_observations = len(Y_pred)

  # Backprop Error; One list element for each layer
  delta_N = []

  # Get Error for Output Layer
  delta = Y_pred - Y_train
  if delta.ndim == 1:
    delta.resize(1, len(delta))
  delta_N.append( delta )

  # Get Error for Hidden Layers working backwards (stop before layer 0; no error in input layer)
  for t in range(T-1,0,-1):
    delta = delta.dot(Theta_L[t][:,1:]) * ( a_N[t][:,1:] * (1 - a_N[t][:,1:]) )
    delta_N.append( delta )
  # Reverse the list so that delta_N[t] is delta that Theta[t] causes on a_N[t+1]
  delta_N.reverse()

  # Calculate Gradient from delta and activation
  # t is the Theta from layer t to layer t+1
  Theta_Gradient_L = []
  for t in range(T):
    Theta_Gradient_L.append( delta_N[t].T.dot(a_N[t]) )

  # Create modified copy of the Theta_L for Regularization
  # Coefficient for theta values from bias unit set to 0 so that bias unit is not regularized
  regTheta = [np.zeros_like(theta) for theta in Theta_L]
  for t, theta in enumerate(Theta_L):
    regTheta[t][:,1:] = theta[:,1:]

  # Average Error + regularization penalty  
  for t in range(T):
    Theta_Gradient_L[t] = Theta_Gradient_L[t] * (1.0/n_observations) + (lmda * regTheta[t])
  
  return Theta_Gradient_L


def fit(X_train, Y_train, Theta_L=[], lmda=1e-5, epochs=2, learning_rate = 1.0, momentum_rate = 0.1, learning_acceleration=1.05, learning_backup=0.5,  X_test=None, Y_test=None):
  """
  fit() calls the predict and back_prop functions for the 
  given number of cycles, tracks error and error improvement rates
    
  Parameters:
    X_train - np.array of training data with dimension n_observations by n_features
    Y_train - np.array of training classes with dimension n_observations by n_classes
    Theta_L - list of theta terms as np.arrays where each theta has dimensions n_units[l+1] by n_units[l]+1
    lmda - regularization term
    epochs -  integer of number of times to update Theta_L
    X_test - np.array of training data with dimension n_observations by n_features
    Y_test - np.array of training classes with dimension n_observations by n_classes
  Returns
    Theta_L - list of theta terms from layer l to l+1
    J_list - list of result of cost function for each epoch
    Learning_rates - list of learning rates used for each epoch
  Notes
    Training and Test data are assumed to be in random order; mini-batch processing does not need to re-randomize
  """
  
  # If no Theta provided, use a 3 layer architecture with hidden_layer units = 2 or y classes or x features
  if not Theta_L:
    hidden_units = max(2, len(Y_train[0]), len(X_train[0]))
    Theta_L = initialize_theta([hidden_units])

  # Initial Learning Rate
  learning_rates = []
  learning_rates.append( learning_rate )

  # Initial Weight Change Terms
  weight_change_L = [np.zeros_like(theta) for theta in Theta_L]
  
  # List of results of cost functions
  J_list = [0] * epochs
  J_test_list = [0] * epochs

  # Initial Forward Pass
  a_N_train, Y_pred = nn_predict(X_train, Theta_L)
  # Initial Cost
  J_list[0] = nn_cost(Y_train, Y_pred)

  # Test Error
  if Y_test is not None:
    a_N_test, Y_pred_test = nn_predict(X_test, Theta_L)
    J_test_list[0] = nn_cost(Y_test, Y_pred_test)
  print 'learning rate: %s' % learning_rate
  print 'momentum rate: %s' % momentum_rate
  print 'learning acceleration: %s' % learning_acceleration
  print 'learning backup: %s' % learning_backup
  
  for i in range(1,epochs):

    # Back Prop to get Theta Gradients
    Theta_grad = back_prop(a_N_train, Y_train, Theta_L, lmda)

    # Update Theta with Momentum
    for l, theta_g in enumerate(Theta_grad):
      weight_change_L[l] = learning_rate * theta_g + (weight_change_L[l] * momentum_rate)
      Theta_L[l] = Theta_L[l] - weight_change_L[l]

    # Update Units
    a_N_train, Y_pred_new = nn_predict(X_train, Theta_L)

    # Check to see if Cost decreased
    J_new = nn_cost(Y_train, Y_pred_new)

    if J_new > J_list[i-1]:
      # Reduce learning rate
      learning_rate = learning_rate * learning_backup
      # Reverse part of adjustment (add back new learning_rate * Theta_grad); Leave momentum in place
      Theta_L = [t + (learning_rate * tg) for t, tg in zip(Theta_L, Theta_grad)]
      # Reduce Prior Weight Change as an approximate fix to Momentum
      weight_change_L = [m * learning_backup for m in weight_change_L]

      a_N_train, Y_pred_new = nn_predict(X_train, Theta_L)
      J_new = nn_cost(Y_train, Y_pred_new)
    else:
      learning_rate = np.min((10,learning_rate * learning_acceleration))

    learning_rates.append(learning_rate)    
    J_list[i] = J_new

    if Y_test is not None:
      a_N_test, Y_pred_test = nn_predict(X_test, Theta_L)
      J_test_list[i] = nn_cost(Y_test, Y_pred_test)
      
  for t, theta in enumerate(Theta_L):
    print 'Theta: %s' % t
    print np.round(theta, 2)

  print 'i:',i,'  - J:',J_list[i]
  print 'i:',i,'  - J test:',J_test_list[i]
    
  return Theta_L, J_list, learning_rates, J_test_list


def translate_to_binary_array(target):
  n_obs = len(target)
  unique_targets = np.unique(target)
  n_unique_targets = len(np.unique(target))

  # Translation of target values to array indicies
  target_translation = dict(zip(unique_targets, range(n_unique_targets)))

  # Create initial target array with all zeros
  target_array = np.zeros((n_obs, n_unique_targets))
  
  # Set 1 value
  for i, val in enumerate(target):
    target_array[i][target_translation[val]] = 1    

  return target_array


def train_test_split(data_array, target_array, split=.8):
  """
  Split into randomly shuffled train and test sets
  Split on Number of records or Percent of records in the training set
  if split is <= 1 then split is a percent, else split is the number of records
  """

  n_obs = len(data_array)
  
  if split <= 1:
    train_len = int(split * n_obs)
  else:
    train_len = int(np.round(split))

  shuffled_index = range(n_obs)
  np.random.shuffle(shuffled_index)

  train_data = data_array[shuffled_index[:train_len]]
  test_data = data_array[shuffled_index[train_len:]]

  train_target = target_array[shuffled_index[:train_len]]
  test_target = target_array[shuffled_index[train_len:]]
  
  print
  print 'Data Set: %d Observations, %d Features' % (data_array.shape[0], data_array.shape[1])
  print 'Training Set: %d Observations, %d Features' % (train_data.shape[0], train_data.shape[1])
  print 'Test Set: %d Observations, %d Features' % (test_data.shape[0], test_data.shape[1])
  print
  print 'Target Set: %d Observations, %d Classes' % (target_array.shape[0], target_array.shape[1])
  print 'Training Set: %d Observations, %d Features' % (train_target.shape[0], train_target.shape[1])
  print 'Test Set: %d Observations, %d Features' % (test_target.shape[0], test_target.shape[1])
  print
  
  return train_data, test_data, train_target, test_target


def nn_test(data_train, target_train, hidden_unit_length_list, epochs, learning_rate, momentum_rate, learning_acceleration, learning_backup, data_test=None, target_test=None):
  
  # Initialize Theta based on selected architecture
  Theta_L = initialize_theta(data_train.shape[1], target_train.shape[1], hidden_unit_length_list)
  
  # Fit
  Theta_L, J_list, learning_rates, J_test_list = fit(data_train, target_train, Theta_L, 1e-5, epochs, learning_rate, momentum_rate, learning_acceleration, learning_backup, data_test, target_test)
  
  # Predict
  a_N, Y_pred = nn_predict(data_test, Theta_L)
  print 'Given X:'
  print data_test[:5]
  print 'Actual Y, Predicted Y:'
  for p in zip(target_test[:10], np.round(Y_pred[:10],3)):
    print p
  print
  print 'CE on Test Set'
  print nn_cost(target_test , Y_pred)

  return target_test, Y_pred, J_list, J_test_list, learning_rates, Theta_L
  

def iris_test(hidden_unit_length_list = [], epochs=2500, learning_rate=0.5, momentum_rate=0.1, learning_acceleration=1.05, learning_backup=0.5):
  data_set = load_iris()

  data = data_set.data
  target = translate_to_binary_array(data_set.target)
  
  # Split into train, test sets
  data_train, data_test, target_train, target_test = train_test_split(data, target, .75)
  
  return nn_test(data_train, target_train, hidden_unit_length_list, epochs, learning_rate, momentum_rate, learning_acceleration, learning_backup, data_test, target_test)

def digits_test(hidden_unit_length_list = [], epochs=2500, learning_rate=0.5, momentum_rate=0.1, learning_acceleration=1.05, learning_backup=0.5):
  data_set = load_digits()

  data = data_set.data
  target = translate_to_binary_array(data_set.target)
  
  # Split into train, test sets
  data_train, data_test, target_train, target_test = train_test_split(data, target, .75)
  
  return nn_test(data_train, target_train, hidden_unit_length_list, epochs, learning_rate, momentum_rate, learning_acceleration, learning_backup, data_test, target_test)


def XOR_test(hidden_unit_length_list = [], epochs=2500, learning_rate=0.5, momentum_rate=0.1, learning_acceleration=1.05, learning_backup=0.5):
  """
  XOR_test is a simple test of the nn printing the predicted value to std out
  Trains on a sample XOR data set
  Predicts a single value
  Accepts an option parameter to set architecture of hidden layers
  """
  
  # Set Data for XOR Test  
  data_train = np.zeros((4,2))
  data_train[0,0] = 1.
  data_train[1,1] = 1.
  data_train[2,:] = 1.
  data_train[3,:] = 0.

  target_train = np.array([1.,1.,0.,0.]).reshape(4,1) 	# Single Class

  # Test X and Y
  data_test = np.array([[1,0],[0,1],[1,1],[0,0]])
  target_test = np.array([[1],[1],[0],[0]])

  print 'Training Data: X & Y'
  print data_train
  print target_train

  return nn_test(data_train, target_train, hidden_unit_length_list, epochs, learning_rate, momentum_rate, learning_acceleration, learning_backup, data_test, target_test)


if __name__ == '__main__':
  """
  Run XOR_test
  Plot Error terms to see learning progress of NN
  """
  target_test, Y_pred, J_list, J_test_list, Theta_L, learning_rates =  XOR_test([2], 300)


