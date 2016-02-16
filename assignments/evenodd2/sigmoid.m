function S = sigmoid(z)
  S = 1 ./ (1 + exp(-z));
end