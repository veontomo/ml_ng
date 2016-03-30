%% Calculates the cost function of the neural network with given 
%% architecture
%% X - training set (each row corresponds to a separate training set).
%%     It is a matrix A x M, A - # training examples, M - # features
%% Y - output matrix A x L, where A is the number of rows of X,
%%     while L is a value of the last element of the row vector defining the 
%%     network architecture. This matrix MUST CONTAIN only 0's and 1's.
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
  %%% what orientation to use when transforming a row vector into a matrix and
  %%% viceverse
  orientation = "v";
  inputNum = size(X, 1);
  layerNum = size(layers, 2);
  %% restore the weight matrices from the row vector
  [weightsMatrices lengths] = formMatrices(weights, layers, orientation);
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
  J = 0;
  for a = 1:inputNum
    %% feedforward: calculate the activations
    prevLayerSize = layers(1);
    A(1, 1) = [1; X(a, :)']; %% it is a column
    for j = 2:layerNum
      Z(1, j) = weightsMatrices{1, j-1} * A{1, j-1};
      A(1, j) = [1; activationFn(Z{1, j})];
    endfor
    Ya = A{1, layerNum}(2:end); %% it is a column
    % deltaJ = - Y(a, :) * log(Ya) - (1-Y(a, :)) * log(1 - Ya);
    %% alternative formulation: in order to avoid expressions like 0 * log(0)
    %% that result in a NaN-value, split the contributions for zero and unity labels
    zeroPos = find(Y(a, :) == 0); %% positions of zero labels
    onePos = find(Y(a, :) == 1); %% positions of unity labels
    deltaJ = - sum(log(Ya(onePos))) - sum(log(1 - (Ya(zeroPos))));
   
    if (isnan(deltaJ) || isinf(deltaJ))
      printf("\niteration %u\n", a);
      printf("J = %2.2f + %2.2f\n", J, deltaJ);
      printf("\n actual label =\n");
      printf("%2.2f, ", Y(a, :));
      printf("\n predicted label =\n");
      printf("%2.2f, ", Ya);
      printf("\nThetas; ");
      printf("%2.2f, ", weights);
      printf("\nExiting...\n");
      printf("\n");
      error("Contribution to the cost function is invalid.");
    endif;
    J = J + deltaJ;

    
    %% backpropagation: calculate the derivatives of the cost function w.r.t. weights
    delta = A{1, layerNum}(2:end)' - Y(a, :); %% it is a row vector
   
    %% highest component of the Q-vector
    %% NB: zero component (that is the lowest value of the second index) 
    %% of the weight matrix does not contribute to the Q-vector
    gradientMatrices(1, layerNum - 1) = gradientMatrices{1, layerNum - 1} + ...
                                       (delta .* A{1, layerNum - 1})';
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
  %J = (J + 1/2 * lambda * (weights * weights'))/inputNum;
  
  
  %% unroll the gradient matrices: pay attention to the orientation (vertical or horizontal)
  grad = [];
  for j = 1: (layerNum-1)
    w = gradientMatrices{1, j};
    %% bias unit weights do not contribute
    weightNoBias = [zeros(size(weightsMatrices{1, j}, 1), 1), weightsMatrices{1, j}(:, 2:end)];
    w = w + lambda*weightNoBias;
    if not(orientation == "v")
      w = w';
    end;
    grad = [grad, w(:)'];
    wTmp = weightNoBias(:);
    J = J + lambda * wTmp' * wTmp/2;
  endfor;
  %% normalize the gradient and the cost function
  J = J/inputNum;
  grad = grad/inputNum;
end