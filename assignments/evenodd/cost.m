function [J grad] = cost(x, y, hidden_layer_size, theta, lambda)

Theta1 = reshape(theta(1:(2*hidden_layer_size)), hidden_layer_size, 2);
Theta2 = reshape(theta((1 + 2*hidden_layer_size):end), 1, hidden_layer_size + 1);

m = size(x, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m, 1) x];
a2 = sigmoid(X * Theta1');
A2 = [ones(size(a2, 1), 1) a2];
h = sigmoid(A2 * Theta2');
reg = trace(Theta1 * Theta1') + trace(Theta2 * Theta2') - trace(Theta1(:, 1) * Theta1(:, 1)') - trace(Theta2(:, 1) * Theta2(:, 1)');

J = -1/m * (trace(y' * log(h)) + trace((1 - y)' * log(1-h) )) + lambda/(2*m)*reg;



Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
for t=1:m
% X has already been augmented with 1 at the beginning
  a1 = x(t);
  A1 = [1; a1];
  z2 = Theta1 * A1;
  a2 = sigmoid(z2);
  A2 = [1; a2];
  z3 = Theta2 * A2;
  a3 = sigmoid(z3);
  delta3 = a3 - y(t);
  tmp = Theta2' * delta3;
  delta2 = tmp(2:end) .* sigmoidGradient(z2);
  Delta1 = Delta1 + delta2 * A1';
  Delta2 = Delta2 + delta3 * A2';
endfor
Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



reg1 = Theta1;
reg2 = Theta2;
reg1(:, 1) = 0;
reg2(:, 1) = 0;

Theta1_grad = Theta1_grad + lambda/m * reg1;
Theta2_grad = Theta2_grad + lambda/m * reg2;




grad = [Theta1_grad(:) ; Theta2_grad(:)];
end