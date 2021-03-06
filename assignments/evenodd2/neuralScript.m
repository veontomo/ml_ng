clc; clear;


archit = [2 5 6 3];
X = [3 2; 1 0; -2 1];
Y = [1 0 0; 0 1 0; 0 0 1];

totParam = archit(2:end) * (1 + archit(1:end-1))' %% the number of weight parameters
                                                  %% that the network must have

%% General
%% params = 2*unifrnd(-1, 1, 1, totParam) - 1;

%% [2 5 6 1]
%% params = [-2.284379  -2.956844  -1.542724   0.901871   0.923262  -0.836405  -2.164785   0.282160  -1.845458  -1.428015  -0.037830  -1.750552   0.657062   0.668678  -1.924558  -1.729429  -2.935313  -1.259305  -1.356328  -0.545008  -0.522996   0.840329  -0.116977   0.306879  -2.336361   0.925126  -2.371843   0.504021  -2.507336  -1.453054   0.025633  -0.726132   0.163484  -2.802077   0.540979  -2.483791  -1.914264  -1.892367  -1.849365  -2.808923  -2.945078  -1.195553   0.246424  -2.835377  -2.819210  -1.299713  -1.741004   0.447743  -2.472249  -0.627677  -2.243185  -0.306001  -2.264560  -1.443901  -2.318978  -2.170291   0.139075   0.710138];

%% [2 5 6 3]
params = [0.0342250  -0.4542839  -2.6528812   0.2030298  -0.6862118  -2.4010993   0.9658549   0.2393028  -0.4567065  -1.9676508  -2.4847111  -0.5784496   0.8370951   0.7565251  -2.0589142  -1.1434968 -2.7254714  -0.6940123  -2.0796824  -1.0235599  -0.9798576  -1.6744904  -0.4115589  -1.7588500  -2.9772614  -0.6937645  -2.0625809  -2.8347321  -0.0015962   0.0044307  -2.8260281  -1.6520862 -0.2487677  -0.8337728  -0.6374790  -1.8227457  -1.1440733  -1.6668174  -2.0439432  -2.1060973   0.2822872  -2.2101957  -1.3606613  -0.8578750   0.3521685  -0.5935213  -0.8815364  -2.1519084 0.6735250   0.0852290   0.7672503   0.2899040   0.8225521  -0.8840523   0.3035542  -0.8017823  -0.4121584  -2.6511934   0.5418102   0.3369430  -1.6006549  -2.5633495   0.7937077  -2.5154720 -0.5368410  -1.8551435  -2.8292586  -0.3567077  -1.0343471  -0.2689843  -1.7131932  -1.3115171];
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

clc; clear;

trainingSetSize = 20;
X = loadData("train-images.idx3-ubyte", trainingSetSize);
Y = loadLabels("train-labels.idx1-ubyte", trainingSetSize);

X = X(trainingSize, 1:inputSize);

archit = [inputSize 20 10];
params = initializeWeights(archit);

generalNNCost = @(Theta)(neuralCost(X, Y, Theta, archit, 5));


[J grad] = generalNNCost(params);


options = optimset('MaxIter', 10);
[theta1, J, exit_flag] = fminunc(generalNNCost, params, options);

