X = [1 2; 1 4; 1 5; 1 1; 1 9];
Y = [1 1 0 0 0]';
theta_init = [0 0];


% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 1000);

% Optimize
[theta, J, exit_flag] = fminunc(@(Theta)(cost(X, Y, Theta)), theta_init, options);

tx = linspace (-2, 2, 4)';
ty = linspace (-1, 1, 3)';
[xx, yy] = meshgrid (tx, ty);
r = sqrt (xx .^ 2 + yy .^ 2) + eps;
tz = sin (r) ./ r;
mesh (tx, ty, tz);