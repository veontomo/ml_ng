function [J grad] = hat(X)
  J = (X' * X - 1)^2;
  grad = 2 * (X' * X - 1) * X;
end