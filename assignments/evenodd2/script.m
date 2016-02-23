%%% Generate training examples
%%% The training set consists of pair-wise different integer numbers
clc;clear;
A = 10000; % the number of the training examples
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
trainingSize = 1000; % the number of examples to train on
testSize = 200; % the number of test examples


%%% single-parameter model
X = [ones(A, 1), DataNorm];

%%% Training the model
[Jtraining Jtest Fscore theta] = profileWRTInputSize(X, Y, trainingSize, testSize, 0);
displayFlow(Jtraining, Jtest, Fscore);

[Jtraining Jtest Fscore theta] = profileWRTInputSize(X, Y, trainingSize, testSize, 10);
displayFlow(Jtraining, Jtest, Fscore);

%%%%%%%%%%%%% another model
X = [ones(A, 1), DataNorm, mod(Data, 2)];
[Jtraining Jtest Fscore theta] = profileWRTInputSize(X, Y, trainingSize, testSize, 0);
displayFlow(Jtraining, Jtest, Fscore);


%%%%  selecting lambda
options = optimset('GradObj', 'on', 'MaxIter', 400);
Xtraining = X(1:trainingSize, :);
Ytraining = Y(1:trainingSize);
Xtest = X((trainingSize+1):(trainingSize + testSize), :);
Ytest = Y((trainingSize+1):(trainingSize + testSize));
theta_init = unifrnd(0, 1, 1, size(Xtraining, 2));
lambdaPool = 0:10:1000;
Jlambda = zeros(size(lambdaPool));
for i = 1: size(lambdaPool, 2)
  lambda = lambdaPool(1, i);
  [theta, J, exit_flag] = fminunc(@(Theta)(cost(Xtraining, Ytraining, Theta, lambda)), theta_init, options);    
  J2 = cost(Xtest, Ytest, theta, lambda);
  Jlambda(i) = J2;
endfor;
plot(lambdaPool, Jlambda, 'color', 'k')
xlabel("lambda");
ylabel("cost function");



