X = [3 2; 1 0; -2 1];
Y = [1; 0; 1];
archit = [2 5 6 1];
totParam = archit(2:end) * (1 + archit(1:end-1))' %% the number of weight parameters
                                                  %% that the network must have
params = 2*randn(1, totParam) - 1;


[J grad] = neuralCost(X, Y, params, archit, 5)

options = optimset('GradObj', 'on', 'MaxIter', 5000);
[theta1, J, exit_flag] = fminunc(@(Theta)(neuralCost(X, Y, Theta, archit, 5)), params, options);

[J1 grad1] = neuralCost(X, Y, params, archit);
eps = 0.000001;

diff = zeros(1, totParam);
for i = 1:totParam
  weights2 = params;
  weights2(i) = weights2(i) + eps;
  [J2 grad2] = neuralCost(X, Y, weights2, archit);
  gradNumeric = (J2 - J1) ./ eps;
  diff(i) = (grad1(i) - gradNumeric);
endfor;
diff


formMatrices(diff, archit)

X = loadData("train-images.idx3-ubyte", 20);
Y = loadLabels("train-labels.idx1-ubyte", 20);

archit = [784 15 10];
totParam = archit(2:end) * (1 + archit(1:end-1))' %% the number of weight parameters
                                                  %% that the network must have
params = 2*randn(1, totParam) - 1;


[J grad] = neuralCost(X, Y, params, archit, 5)

options = optimset('GradObj', 'on', 'MaxIter', 5000);
[theta1, J, exit_flag] = fminunc(@(Theta)(neuralCost(X, Y, Theta, archit, 5)), params, options);

