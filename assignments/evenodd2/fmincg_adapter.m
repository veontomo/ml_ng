%% A simple adapter for fmincg function.
%% fmincg expects that its argument defining the neural network weights is a
%% column vector, while the same argument in function "neuralCost" is a
%% row vector.
%% X, Y, weigthColumn, archit, lambda - are the same as in "neuralCost" function,
%% except for weigthColumn which is a row, not a column.
%% Returns:
%% the same output as "neuralCost" produces except for the second return value
%% that is transposed before being returned.
function [J gradColumn] = fmincg_adapter(X, Y, weigthColumn, archit, lambda)
  [J gradRow] = neuralCost(X, Y, weigthColumn', archit, lambda);
  gradColumn = gradRow';
end