clear ; close all; clc;

addpath('former_ex4');

input_layer_size  = 5; 
hidden_layer_size = 50;
num_labels = 4;
trainingSetSize = 15;

archit = [input_layer_size hidden_layer_size num_labels];

%% =========== Generating Training Set Data =============
X = cos((1:trainingSetSize)' .* (1:input_layer_size));
Y = repmat([1 zeros(1, input_layer_size)], 1, ceil(trainingSetSize /(1 + input_layer_size)))(1:trainingSetSize)';
m = size(X, 1);
lambda = 10;

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
options = optimset('MaxIter', 400);

[nn_params1, cost1] = fminunc(specificNNCost, Theta', options);
[nn_params2, cost2] = fminunc(generalNNCost, Theta, options);

[nn_params1, cost1] = fmincg(specificNNCost, Theta', options);
[nn_params2, cost2] = fmincg(generalNNCost, Theta, options);
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
