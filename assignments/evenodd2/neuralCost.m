%% Calculates the cost function of the neural network with given 
%% architecture
%% X - training set (each row corresponds to a separate training set).
%%     It is a matrix A x M, A - # training examples, M - # features
%% Y - output. It is a vector A x 1.
%% weights - list of weights. It is an unfolded version of all weights.
%%         It includes weights for the bias units as well.
%% layers - list of units in each layer.
function [J grad] = neuralCost(X, Y, weights, layers)
  %% # units in the previous layer
  prevLayerSize = layers(1);
  A = [1 X]';
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
  J = A(2:end)
end