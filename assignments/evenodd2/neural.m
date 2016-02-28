%% returns minimal value of the cost function and corresponding value of the
%% weight parameters (in unfolded form) of a neural network which architecture
%% is defined by the first parameter
%% layers - list of integers defining the number of units in the network layers.
%%          The first element is the number of the units in the input layer,
%%          the second element is the number of the units in the first hidden layer,
%%          the last element is the number of the units in the output layer. 
%% inputData - training set. Each row corresponds to a separate training set.
%% outputData - output of the training set.
function [J theta] = neural(layers, inputData, outputData)
  % 1. create unfolded version of the parameters
  % 2. cost function 
  % 3. minimize the cost function
end