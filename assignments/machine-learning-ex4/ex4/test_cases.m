% This test case is: 
% https://www.coursera.org/learn/machine-learning/module/Aah2H/discussions/uPd5FJqnEeWWpRIGHRsuuw

% all test cases: 
% https://www.coursera.org/learn/machine-learning/discussions/0SxufTSrEeWPACIACw4G5w

il = 2;              % input layer
hl = 2;              % hidden layer
nl = 4;              % number of labels
nn = [ 1:18 ] / 10;  % nn_params
X = cos([1 2 ; 3 4 ; 5 6]);
y = [4; 2; 3];
lambda = 4;
[J_actual grad_actual] = nnCostFunction(nn, il, hl, nl, X, y, lambda)

J_expected = 19.474
grad_expected = [0.76614; 0.97990; 0.37246;0.49749;0.64174;0.74614;0.88342;0.56876;0.58467;0.59814;1.92598;1.94462;1.98965;2.17855;2.47834;2.50225;2.52644;2.72233];

error_J = abs((1 - J_actual/J_expected));
error_grad = abs((1 - grad_actual ./ grad_expected));

if (error_J < 0.01) disp("The cost function is correct") else printf("The cost function is wrong! Expected = %.5f vs actual = %.5f", J_expected, J_actual); endif

if (error_grad < 0.01) disp("The gradient is correct") else printf("The gradient is wrong!"); endif