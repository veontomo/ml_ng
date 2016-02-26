function [J grad] = cost(X, Y, Theta, lambda)
  h = sigmoid(X * Theta'); % column
  m = size(X, 1);
  J = 1/m*(-sum(log(h(find(Y==1)))) - sum(log(1-h(find(Y==0)))));
  grad = (h - Y)' * X/m;
  if lambda != 0 
    ThetaWithoutFirst = Theta(2:end);
    J = J + lambda/(2*m) * ThetaWithoutFirst * ThetaWithoutFirst';
    grad = grad + lambda/m * [0 ThetaWithoutFirst];
  endif
end