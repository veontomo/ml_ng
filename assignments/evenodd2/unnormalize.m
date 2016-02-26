%% reconstructs a vlue using its normalized value and paramaters mu (mean) 
%% and range (difference btw max and min elements of each column of some set)
function X = unnormalize(Xnorm, mu, range)
  X = Xnorm .* range + mu;
end

