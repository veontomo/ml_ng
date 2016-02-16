 X = [1 2; 1 4; 1 5; 1 1; 1 9];
 Y = [1 1 0 0 0]';
 theta_init = [1 1];

 

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = fminunc(@(Theta)(cost(X, Y, Theta)), theta_init, options);

