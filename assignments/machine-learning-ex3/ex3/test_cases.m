% https://www.coursera.org/learn/machine-learning/discussions/5g8LaZTCEeW0dw6k4EUmPw
% Here are the test cases for ex3 by Tom Mosher:
% input:

clc; 

theta = [-2; -1; 1; 2];
X = [ones(3,1) magic(3)];
y = [1; 0; 1] >= 0.5;       % creates a logical array
lambda = 3;
[J grad] = lrCostFunction(theta, X, y, lambda);
% output:
answer_J = 7.6832;
answer_grad = [0.31722; -0.12768; 2.64812; 4.23787];

printf("Expected value of the cost function: %f\n", answer_J);
printf("Actual value of the cost function: %f\n", J);
if (abs(J - answer_J) < 0.001),
  disp('The cost function is calculated correctly.');
  else 
  disp('The cost function is NOT calculated correctly.');
endif

disp("Expected value of the gradient function:");
printf("%f ", answer_grad);
disp("Actual value of the gradient function:");
printf("%f ", grad);

if (abs(grad - answer_grad) < 0.001),
  disp('The gradient is calculated correctly.');
  else 
  disp('The gradient is NOT calculated correctly.');
endif
