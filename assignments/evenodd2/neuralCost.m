%% Calculates the cost function of the neural network with given 
%% architecture
%% X - training set (each row corresponds to a separate training set).
%%     It is a matrix A x M, A - # training examples, M - # features
%% Y - output. It is a vector A x 1.
%% theta - list of weights. It is an unfolded version of all weights.
%%         It includes weights for the bias units as well.
%% layers - list of units in each layer.
function [J grad] = neuralCost(X, Y, theta, layers)
  %% # units in the previous layer
  prevLayerSize = 0;
  %% # weights that have already been taken into consideration
  counter = 0;
  for layerSize = layers
    length = (prevLayerSize + 1) * layerSize
    thetaTmp = reshape(theta((counter+1):(counter + length)), prevLayerSize + 1, layerSize)
    fprintf("layer size: %2u", layerSize);
    counter = counter + length;
    prevLayerSize = layerSize;
  endfor
end