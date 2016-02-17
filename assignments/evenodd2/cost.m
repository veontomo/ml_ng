function [J grad] = cost(X, Y, Theta)
  h = sigmoid(X * Theta'); % column
  m = size(X, 1);
  %J = (- Y' * log(h) - (1 - Y)' * log(1-h))/m
  J = -sum(log(h(find(Y==1)))) - sum(log(1-h(find(Y==0))));
  J = J/m;
  grad = (h - Y)' * X/m;
end