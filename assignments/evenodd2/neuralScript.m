X = [3 4 1; 1 3 0; -2 5 3];
Y = [1; 0; 1];
params = -1:0.02:1;
archit = [3 3 3 2 1];
[J grad] = neuralCost(X, Y, params, archit)

weightTotal = 20;
archit = [2 3 2 1];
X = [6 7; 3 4];
Y = [1; 0];
weights1 = unifrnd(-1, 1, 1, weightTotal);
[J1 grad1] = neuralCost(X, Y, weights1, archit);

eps = 0.000001;

diff = zeros(1, weightTotal);
for i = 1:weightTotal
  weights2 = weights1;
  weights2(i) = weights2(i) + eps;
  [J2 grad2] = neuralCost(X, Y, weights2, archit);
  gradNumeric = (J2 - J1) ./ eps;
  diff(i) = (grad1(i) - gradNumeric);
endfor;
diff


formMatrices(diff, archit)