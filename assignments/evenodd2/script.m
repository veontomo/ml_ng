A = 1000;
X = unique(round(A*unifrnd(0, 1, A, 1)));
Y = mod(X, 2);
X = [ones(size(X, 1), 1), X];
theta_init = [0 0];


% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

cutoff = round(0.8*size(X, 1)); % the number of examples to track
Jpool = zeros(1, cutoff);
for i = 1:cutoff
  [theta, J, exit_flag] = fminunc(@(Theta)(cost(X(1:i, :), Y(1:i, :), Theta)), theta_init, options);
  Jpool(i) = J;
endfor

plot(1:cutoff, Jpool)
