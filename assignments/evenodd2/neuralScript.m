X = [3; 1; -2];
Y = [1; 0; 1];
archit = [1 1];
totParam = archit(2:end) * (1 + archit(1:end-1))' %% the number of weight parameters
                                                  %% that the network must have
params = 2*randn(1, totParam) - 1;


[J grad] = neuralCost(X, Y, params, archit)

options = optimset('GradObj', 'on', 'MaxIter', 5000);
[theta1, J, exit_flag] = fminunc(@(Theta)(neuralCost(X, Y, Theta, archit)), params, options);

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