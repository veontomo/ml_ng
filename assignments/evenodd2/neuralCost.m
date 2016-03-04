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
  %% set of activations for every layer
  A = cell(size(layers));
  %% initialize the set of gradient matrices
  gradientMatrices = cell(size(weightsMatrices));
  %% recursive object
  Q = cell(size(weightsMatrices));
  for i = 1:size(weightsMatrices, 2)
    gradientMatrices(1, i) = zeros(size(weightsMatrices{1, i}));
    Q(1, i) = zeros(1, layers(i+1));
  endfor;
  Q
  Yproduced = [];
  J = 0;
  inputNum = size(X, 1);
  layerNum = size(layers, 2);
  for a = 1:inputNum
    %% feedforward: calculate the activations
    prevLayerSize = layers(1);
    A(1, 1) = [1; X(a, :)']; %% it is a column
    for j = 2:layerNum
      Z = weightsMatrices{1, j-1} * A{1, j-1}; 
      A(1,j) = [1; activationFn(Z)];
    endfor
    Yproduced = [Yproduced; A{1, layerNum}(2:end)'];
    %% backpropagation: calculate the derivatives of the cost function w.r.t. weights
    delta = A{1, layerNum}(2:end)' - Y(a, :); %% it is a row vector
    gradientMatrices(1, layerNum-1) = gradientMatrices{1, layerNum-1} + delta * A{1, j};
    
    for j = (layerNum-2):-1:1
      %% tmp = delta * A{1, j};
      %%gradientMatrices(1, j) = gradientMatrices{1, j} + tmp;
    endfor;
    
  endfor
  J = (- Y' * log(Yproduced) - (1 - Y') * log(1 - Yproduced))/inputNum;
  gradTmp = (Yproduced - Y)
end