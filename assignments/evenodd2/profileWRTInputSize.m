%% calculates dependence of the cost function and F1 score on the size of the training set.
%% X - matrix of the training set. Each row corresponds to a single training data.
%% Y - column of results corresponding to each training set.
%% trainingSize - how many rows of X to use in order to train the model.
%%                The rows are selected starting from the first row up 
%%                to the row trainingSize.
%% testSize - how many rows of X to use in order to evaluate the method 
%%            effectiveness. The rows are selected starting from (trainingSize+1)
%%            to (trainingSize + testSize).
%% lambda - value of the regularization coefficient in the cost function.
%% The method returns arrays:
%% Jtraining - array containing trainingSize-many values of the cost function
%%             Jtraining(i) is a value of the cost function trained on X(1:i, :)
%% Jtest - array containing trainingSize-many values of the cost function
%%         calculated on X((trainingSize+1):(trainingSize + testSize + 1), :)
%%         with parameters theta taken from the training on the above step
%% Fscore - array containing trainingSize-many values of F1 scores
function [Jtraining Jtest Fscore theta] = profileWRTInputSize(X, Y, trainingSize, testSize, lambda)
  Xtraining = X(1:trainingSize, :);
  Ytraining = Y(1:trainingSize);
  theta = zeros(trainingSize, size(X, 2));
  Xtest = X((trainingSize+1):(trainingSize + testSize), :);
  Ytest = Y((trainingSize+1):(trainingSize + testSize));
  Jtraining = zeros(1, trainingSize);
  Jtest = zeros(1, trainingSize);
  Fscore = zeros(1, trainingSize);
  options = optimset('GradObj', 'on', 'MaxIter', 1000);
  theta_init = unifrnd(0, 1, 1, size(Xtraining, 2));
  for i = 1:trainingSize
    [theta1, J, exit_flag] = fminunc(@(Theta)(cost(Xtraining(1:i, :), Ytraining(1:i, :), Theta, lambda)), theta_init, options);
    J2 = cost(Xtest, Ytest, theta1, 0);
    Jtest(i) = J2;
    Jtraining(i) = J;
    theta(i, :) = theta1;
    %% Method precision, recall 
    Ypredicted = Xtest * theta1' > 0;
    [tp tn fp fn] = classifyPredictions(Ytest, Ypredicted);
    Fscore(i) = calculateF1score(tp, tn, fp, fn);
  endfor
end