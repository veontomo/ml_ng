function D = activationFnDeriv(z)
  D = sigmoid(z) .* (1 - sigmoid(z));
end