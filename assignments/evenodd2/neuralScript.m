archit = [2 5 6 1];
X = [3 2; 1 0; -2 1];
Y = [1; 0; 1];

totParam = archit(2:end) * (1 + archit(1:end-1))' %% the number of weight parameters
                                                  %% that the network must have

%% General
%% params = 2*unifrnd(-1, 1, 1, totParam) - 1;

%% [2 5 6 1]
params = [-2.284379  -2.956844  -1.542724   0.901871   0.923262  -0.836405  -2.164785   0.282160  -1.845458  -1.428015  -0.037830  -1.750552   0.657062   0.668678  -1.924558  -1.729429  -2.935313  -1.259305  -1.356328  -0.545008  -0.522996   0.840329  -0.116977   0.306879  -2.336361   0.925126  -2.371843   0.504021  -2.507336  -1.453054   0.025633  -0.726132   0.163484  -2.802077   0.540979  -2.483791  -1.914264  -1.892367  -1.849365  -2.808923  -2.945078  -1.195553   0.246424  -2.835377  -2.819210  -1.299713  -1.741004   0.447743  -2.472249  -0.627677  -2.243185  -0.306001  -2.264560  -1.443901  -2.318978  -2.170291   0.139075   0.710138];

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

options = optimset('GradObj', 'on', 'MaxIter', 100);
[theta1, J, exit_flag] = fminunc(@(Theta)(neuralCost(X, Y, Theta, archit, 5)), params, options);

