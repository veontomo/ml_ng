 X = [1 2; 1 4; 1 5; 1 1; 1 9];
 Y = [1 1 -1 -1 -1]';
 theta_init = [2 7.6];

 
 
 h = sigmoid(X * Theta') % column
  m = size(X, 1)
  J = (- Y' * log(h) - (1 - Y)' * log(1-h))/m
  grad = (h - Y)' * X/m


% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = fminunc(@(Theta)(cost(X, Y, Theta)), theta_init, options);

