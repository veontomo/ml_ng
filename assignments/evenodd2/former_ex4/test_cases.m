% This test case is: 
% https://www.coursera.org/learn/machine-learning/module/Aah2H/discussions/uPd5FJqnEeWWpRIGHRsuuw

% all test cases: 
% https://www.coursera.org/learn/machine-learning/discussions/0SxufTSrEeWPACIACw4G5w
clc;
il = 2;              % input layer
hl = 2;              % hidden layer
nl = 4;              % number of labels
nn = [ 1:18 ] / 10;  % nn_params
X = cos([1 2 ; 3 4 ; 5 6]);
y = [4; 2; 3];
lambda = 4;
[J_actual grad_actual] = nnCostFunction(nn, il, hl, nl, X, y, lambda)

J_expected = 19.474
grad_expected = [0.766138 0.979897 -0.027540 -0.035844 -0.024929 -0.053862 0.883417 0.568762 0.584668 0.598139 0.459314 0.344618 0.256313 0.311885 0.478337 0.368920 0.259771 0.322331]';

error_J = abs((1 - J_actual/J_expected));
error_grad = abs((1 - grad_actual ./ grad_expected));

if (error_J < 0.01) disp("The cost function is correct") else printf("The cost function is wrong! Expected = %.5f vs actual = %.5f", J_expected, J_actual); endif

if (error_grad < 0.01) disp("The gradient is correct") else printf("The gradient is wrong!"); endif