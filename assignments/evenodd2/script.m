%%% Generate training examples
%%% The training set consists of pair-wise different integer numbers

A = 30; % the number of the training examples
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
X = [ones(size(Data, 1), 1), Data];

theta_init = unifrnd(-2, 2, 1, size(X, 2));


% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 100);

cutoff = 20; % the number of examples to track
Jpool = zeros(1, cutoff);
for i = 1:cutoff
  [theta, J, exit_flag] = fminunc(@(Theta)(cost(X(1:i, :), Y(1:i, :), Theta)), theta_init, options);
  Jpool(i) = J;
endfor

plot(1:cutoff, Jpool)


