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
  %% restore the weight matrices from the row vector
  [weightsMatrices lengths] = formMatrices(weights, layers);
  Yproduced = [];
  J = 0;
  inputNum = size(X, 1);
  layerNum = size(layers, 2);
  for i = 1:inputNum
    prevLayerSize = layers(1);
    A = [1; X(i, :)']; %% it is a column
    for j = 2:layerNum
      Z = weightsMatrices{1, j-1} * A; 
      A = [1; sigmoid(Z)];
    endfor
    Yproduced = [Yproduced; A(2:end)'];
  endfor
  J = (- Y' * log(Yproduced) - (1 - Y') * log(1 - Yproduced))/inputNum;
  gradTmp = (Yproduced - Y)
end