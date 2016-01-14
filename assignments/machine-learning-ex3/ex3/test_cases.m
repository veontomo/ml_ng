% https://www.coursera.org/learn/machine-learning/discussions/5g8LaZTCEeW0dw6k4EUmPw
% Here are the test cases for ex3 by Tom Mosher:
% input:

clc; 


%%%% lrCostFunction

theta = [-2; -1; 1; 2];
X = [ones(3,1) magic(3)];
y = [1; 0; 1] >= 0.5;       % creates a logical array
lambda = 3;
[J grad] = lrCostFunction(theta, X, y, lambda);
% output:
expected_J = 7.6832;
expected_grad = [0.31722; -0.12768; 2.64812; 4.23787];

printf("Expected value of the cost function: %f\n", expected_J);
printf("Actual value of the cost function: %f\n", J);
if (abs(J - expected_J) < 0.001),
  disp('The cost function is calculated correctly.');
  else 
  disp('The cost function is NOT calculated correctly.');
endif

disp("Expected value of the gradient function:");
printf("%f ", expected_grad);
disp("Actual value of the gradient function:");
printf("%f ", grad);

if (abs(grad - expected_grad) < 0.001),
  disp('The gradient is calculated correctly.');
  else 
  disp('The gradient is NOT calculated correctly.');
endif


%%%% oneVsAll 

%input:
clc;
X = [magic(3) ; sin(1:3); cos(1:3)];
y = [1; 2; 2; 1; 3];
num_labels = 3;
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);
%output:
expected_all_theta = [ -0.559478   0.619220  -0.550361  -0.093502;
  -5.472920  -0.471565   1.261046   0.634767;
   0.068368  -0.375582  -1.652262  -1.410138]

if (abs(expected_all_theta - all_theta) < 0.001),
  disp('The training params are calculated correctly.');
  else 
  disp('The training params NOT calculated correctly.');
endif