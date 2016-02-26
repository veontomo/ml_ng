%% normalizes each column of set X
%% returns:
%% mu - a row containing mean of each column of X
%% range - a row containing difference btw max and min elements of each column of X
%% Xnorm - normalized version of X (each column of X gets subtracted by 
%% corresponding value of mu and then divided by corresponding value of range)
function [mu range Xnorm] = normalize(X)
 mu = mean(X);
 range = max(X) - min(X);
 Xnorm = (X - mu) ./ repmat(range, size(X, 1), 1);
end

