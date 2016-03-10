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
%% lambda - regularization parameter
%% Returns:
%% J - value of the cost function for given input
%% grad - derivatives of the cost function w.r.t the weight parameters at given
%%        point.
function [J grad] = neuralCost(X, Y, weights, layers, lambda)
  inputNum = size(X, 1);
  layerNum = size(layers, 2);
  %% restore the weight matrices from the row vector
  [weightsMatrices lengths] = formMatrices(weights, layers);
  %% set of activations for every layer
  A = cell(1, layerNum);
  Z = cell(1, layerNum);
  %% initialize the set of gradient matrices
  gradientMatrices = cell(size(weightsMatrices));
  for i = 1:size(weightsMatrices, 2)
    gradientMatrices(1, i) = zeros(size(weightsMatrices{1, i}));
  endfor;
  %% Set of Q-vectors. 
  %% The number of Q-vectors is equal to the total number of the network 
  %% layers minus 2.
  Q = cell(1, layerNum - 2);
  for k = 1:(layerNum - 2)
    Q(1, k) = zeros(1, layers(k+1));
  endfor;
  Yproduced = [];
  J = 0;
  for a = 1:inputNum
    %% feedforward: calculate the activations
    prevLayerSize = layers(1);
    A(1, 1) = [1; X(a, :)']; %% it is a column
    for j = 2:layerNum
      Z(1, j) = weightsMatrices{1, j-1} * A{1, j-1}; 
      A(1, j) = [1; activationFn(Z{1, j})];
    endfor
    Yproduced = [Yproduced; A{1, layerNum}(2:end)'];
    fprintf("\nY(%u) = \n", a);
    Y(a, :)
    fprintf("\nYproduced(%u) = \n", a);
    Yproduced(a, :)
    J = J + (- Y(a, :) * log(Yproduced(a, :)') - (1-Y(a, :)) * log(1 - Yproduced(a, :)'));
    fprintf("\nJ = \n");
    J
    
    
    %% backpropagation: calculate the derivatives of the cost function w.r.t. weights
    delta = A{1, layerNum}(2:end)' - Y(a, :); %% it is a row vector
   
    %% highest component of the Q-vector
    %% NB: zero component (that is the lowest value of the second index) 
    %% of the weight matrix does not contribute to the Q-vector
    gradientMatrices(1, layerNum - 1) = gradientMatrices{1, layerNum - 1} + (delta .* A{1, layerNum - 1})';
    if layerNum > 2 
      Q(1, layerNum - 2) = delta * weightsMatrices{1, layerNum - 1}(:, 2:end);
      tmp = Q{1, layerNum - 2} .* activationFnDeriv(Z{1, layerNum-1}');
      gradientMatrices(1, layerNum - 2) = gradientMatrices{1, layerNum - 2} + (A{1, layerNum - 2} * tmp)';
      for j = (layerNum-3):-1:1
        Q(1, j) = (Q{1, j+1} .* activationFnDeriv(Z{1, j+2}')) * weightsMatrices{1, j+1}(:, 2:end);
        tmp = (Q{1, j} .* activationFnDeriv(Z{1, j+1}')) .* A{1, j};
        gradientMatrices(1, j) = gradientMatrices{1, j} + tmp';
      endfor;
    endif;
    
  endfor
  %J = (- Y' * log(Yproduced) - (1 - Y') * log(1 - Yproduced) + 1/2 * lambda * (weights * weights'))/inputNum;
  J = (J + 1/2 * lambda * (weights * weights'))/inputNum;
  
  %% unroll the gradient matrices.
  %% NB: there are two transpositions of the gradient matrix:
  %% 1. the unrolling (:) goes over columns, while we need it over rows. 
  %% 2. the result of the unrolling is a row vector, while we need a column one.
  grad = [];
  for j = 1: (layerNum-1)
    grad = [grad, gradientMatrices{1, j}'(:)'];
  endfor;
  %% normalize the gradient
  grad = (grad + lambda * weights) /inputNum;
end