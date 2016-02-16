function [J grad] = cost(X, Y, Theta)
%  fprintf("X = ");
%  fprintf("%u ", X(:, 2));
%  fprintf("\nTheta = ");
%  fprintf("%2.4f ", Theta);
%  fprintf("\n");
  h = sigmoid(X * Theta'); % column
  m = size(X, 1);
  J = (- Y' * log(h) - (1 - Y)' * log(1-h))/m;
  grad = (h - Y)' * X/m;
end