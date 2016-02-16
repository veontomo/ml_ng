function J = cost(X, Y, Theta)
  h = sigmoid(X * Theta'); % column
  J = - Y' * log(h) - (1 - Y)' * log(1-h);
end