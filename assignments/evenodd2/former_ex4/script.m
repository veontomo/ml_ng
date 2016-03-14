clear ; close all; clc

input_layer_size  = 20*20; 
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
archit = [input_layer_size hidden_layer_size num_labels];

%% =========== Loading and Visualizing Data =============
trainingSetSize = 100;
X = loadData("train-images.idx3-ubyte", trainingSetSize);
X = X(:, 1:input_layer_size);
Y = loadLabels("train-labels.idx1-ubyte", trainingSetSize);
m = size(X, 1);

%% ================ Initializing Pameters ================

initial_nn_params = initializeWeights(archit);

%% ================ short hand definitions ================
specificNNCost = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, Y, 1);
generalNNCost = @(p) neuralCost(X, Y, p, archit, 1);

%% ===================  Testing cost function ===================


[J grad] = specificNNCost(initial_nn_params);

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
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

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
