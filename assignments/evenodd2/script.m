%%% Generate training examples
%%% The training set consists of pair-wise different integer numbers
clc;clear;
A = 1000; % the number of the training examples
Data = zeros(A, 1);
maxLoopIter = 3;  % the maximal number of iterations to pick up a random 
                  % integer number before widening the range
rightBound = A;   % the range (0, rightBound) from which a random
                  % integer is to be picked up. The rightBound increases
                  % if a new value can not be added after maxLoopIter iterations.

for i=1:A
  loopCounter = 0;
  do
    if loopCounter > maxLoopIter
      rightBound = 10*rightBound;
      loopCounter = 0;
    endif;
    tmp = round(unifrnd(0, rightBound)); 
    loopCounter = loopCounter + 1;
  until !ismember(tmp, Data(1:(i-1)))
  Data(i) = tmp;
endfor;

Y = mod(Data, 2);


[mu range DataNorm] = normalize(Data);
trainingSize = 500; % the number of examples to train on
testSize = 100; % the number of test examples


%%% single-parameter model
X = [ones(A, 1), DataNorm];

% Training the model

displayFlow(X, Y, trainingSize, testSize, 0);

options = optimset('GradObj', 'on', 'MaxIter', 400);
theta_init = unifrnd(0, 1, 1, size(X, 2));

Xtraining = X(1:trainingSize, :);
Ytraining = Y(1:trainingSize);
Xtest = X((trainingSize+1):(trainingSize + testSize), :);
Ytest = Y((trainingSize+1):(trainingSize + testSize));


Jtraining = zeros(1, trainingSize);
Jtest = zeros(1, trainingSize);
Fscore = zeros(1, trainingSize);


for i = 1:trainingSize
  [theta, J, exit_flag] = fminunc(@(Theta)(cost(Xtraining(1:i, :), Ytraining(1:i, :), Theta, 10)), theta_init, options);
  J2 = cost(Xtest, Ytest, theta, 0);
  Jtest(i) = J2;
  Jtraining(i) = J;
  %% Method precision, recall and accuracy
  Ypredicted = Xtest * theta' > 0;
  [tp tn fp fn] = classifyPredictions(Ytest, Ypredicted);
  Prec = tp/(tp + fp);
  Rec = tp/(tp + fn);
  Acc = (tp + tn)/(tp + tn + fn + fp);
  Fscore(i) = 2*Prec*Rec/(Prec + Rec);
endfor
subplot (2, 1, 1)
hold on;
plot(1:trainingSize, Jtraining, 'color', 'r')
plot(1:trainingSize, Jtest, 'color', 'k')
hold off;
subplot (2, 1, 2)
plot(1:trainingSize, Fscore, 'color', 'b')



%%%%%%%%%%%%% another model
X = [ones(A, 1), DataNorm, mod(Data, 2)];
theta_init = unifrnd(-2, 2, 1, size(X, 2));


Xtraining = X(1:trainingSize, :);
Ytraining = Y(1:trainingSize);
Xtest = X((trainingSize+1):(trainingSize + testSize), :);
Ytest = Y((trainingSize+1):(trainingSize + testSize));


Jtraining = zeros(1, trainingSize);
Jtest = zeros(1, trainingSize);
Fscore = zeros(1, trainingSize);
for i = 1:trainingSize
  [theta, J, exit_flag] = fminunc(@(Theta)(cost(Xtraining(1:i, :), Ytraining(1:i, :), Theta, 3)), theta_init, options);
  J2 = cost(Xtest, Ytest, theta, 0);
  Jtest(i) = J2;
  Jtraining(i) = J;
  %% Method precision, recall and accuracy
  Ypredicted = Xtest * theta' > 0;
  [tp tn fp fn] = classifyPredictions(Ytest, Ypredicted);
  Prec = tp/(tp + fp);
  Rec = tp/(tp + fn);
  Acc = (tp + tn)/(tp + tn + fn + fp);
  Fscore(i) = 2*Prec*Rec/(Prec + Rec);

endfor

subplot (2, 1, 1)
hold on;
plot(1:trainingSize, Jtraining, 'color', 'r')
plot(1:trainingSize, Jtest, 'color', 'k')
hold off;
subplot (2, 1, 2)
plot(1:trainingSize, Fscore, 'color', 'b')


