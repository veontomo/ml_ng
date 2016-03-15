clear ; close all; clc

addpath('former_ex4');

input_layer_size  = 2; 
hidden_layer_size = 2;
num_labels = 4;
               
archit = [input_layer_size hidden_layer_size num_labels];

%% =========== Loading and Visualizing Data =============
X = cos([1 2]);
Y = [0 0 0 1];
m = size(X, 1);
lambda = 0;

%% ================ Initializing Parameters ================

Theta = initializeWeights(archit, false);

%% ================ short hand definitions ================
specificNNCost = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda);
generalNNCost = @(p) neuralCost(X, Y, p, archit, lambda);

%% ===================  Testing cost function ===================
[J1 grad1] = specificNNCost(Theta);
[J2 grad2] = generalNNCost(Theta);

%% ===================  Training NN ===================
%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, Theta, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%% ================= Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

displayData(Theta1(:, 2:end));
displayData(Theta2(:, 2:end));

%% ================= Predict =================
Y = y * (1:num_labels)';
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y)) * 100);
