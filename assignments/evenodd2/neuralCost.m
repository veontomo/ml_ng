%% Calculates the cost function of the neural network with given 
%% architecture
%% X - training set (each row corresponds to a separate training set).
%%     It is a matrix A x M, A - # training examples, M - # features
%% Y - output matrix A x L, where A is the number of rows of X,
%%     while L is a value of the last element of the row vector defining the 
%%     network architecture.
%% weights - row vector of weights. It is an unfolded version of all weights.
%%         It includes weights for the bias units as well.
%% layers - row vector that defines the network architecture. Its elements are 
%%         the number of units in corresponding layers.
function [J grad] = neuralCost(X, Y, weights, layers)
  %% calculates the number of all weights that a neural network with a given
  %% architecture should have
  requiredWeightNum = layers(2:end) * (1 + layers(1:end-1))';
  weightsSize = size(weights, 2);
  if (requiredWeightNum != weightsSize) 
    printf("Warning: the network should have %u weights, while %u are given.\n", requiredWeightNum, weightsSize);
  endif;
  %% output produced by the network
  Yproduced = [];
  J = 0;
  %% for-loop iterates over the columns, while training examples 
  %% are arranged in rows. This is the reason for the transposition operation.
  for x = X'
    %% # units in the previous layer
    prevLayerSize = layers(1);
    A = [1; x];
    %% # weights that have already been taken into consideration
    counter = 0;
    for layerSize = layers(2:end)
      length = (prevLayerSize + 1) * layerSize;
      layerWeights = reshape(weights((counter+1):(counter + length)), prevLayerSize + 1, layerSize)';
      Z = layerWeights * A;
      A = [1; sigmoid(Z)];
      counter = counter + length;
      prevLayerSize = layerSize;
    endfor
    %% append what the output units produce
    Yproduced = [Yproduced; A(2:end)'];
  endfor;
  J = (- Y' * log(Yproduced) - (1 - Y') * log(1 - Yproduced))/size(Y, 1);
end