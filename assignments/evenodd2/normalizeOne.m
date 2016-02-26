%% Normalizes X using parameters.
%% mu - has a meaning of a mean value (of some other set)
%% range - has a mening of a difference btw  a max and a min elements of some other set
function Xnorm = normalizeOne(X, mu, range)
 Xnorm = (X - mu) ./ repmat(range, size(X, 1), 1);
end

