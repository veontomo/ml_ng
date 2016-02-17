%%% Generate training examples
%%% The training set consists of pair-wise different integer numbers
clc;clear;
A = 4000; % the number of the training examples
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

%%% single-parameter model
[mu range DataNorm] = normalize(Data);
X = [ones(size(Data, 1), 1), DataNorm];


% Training the model
options = optimset('GradObj', 'on', 'MaxIter', 400);
theta_init = unifrnd(0, 1, 1, size(X, 2));

trainingSize = 2000; % the number of examples to train on
Jpool = zeros(1, trainingSize);
for i = 5:trainingSize
  [theta, J, exit_flag] = fminunc(@(Theta)(cost(X(1:i, :), Y(1:i, :), Theta)), theta_init, options);
  Jpool(i) = J;
endfor

plot(1:trainingSize, Jpool)

%%% Method precision, recall and accuracy
printf("theta = ");
theta
yPredicted = X * theta' > 0;
[tp tn fp fn] = classifyPredictions(Y, yPredicted)

Prec = tp/(tp + fp)
Rec = tp/(tp + fn)
Acc = (tp + tn)/(tp + tn + fn + fp)
Fscore = 2*Prec*Rec/(Prec + Rec)



%%%%%%%%%%%%% another model
X = [ones(size(Data, 1), 1), DataNorm, mod(Data, 2)];
theta_init = unifrnd(-2, 2, 1, size(X, 2));


% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

trainingSize = 500; % the number of examples to train on
Jpool = zeros(1, trainingSize);
for i = 1:trainingSize
  [theta, J, exit_flag] = fminunc(@(Theta)(cost(X(1:i, :), Y(1:i, :), Theta)), theta_init, options);
  Jpool(i) = J;
endfor

plot(1:trainingSize, Jpool)


yPredicted = X * theta' > 0;
[tp tn fp fn] = classifyPredictions(Y, yPredicted)

