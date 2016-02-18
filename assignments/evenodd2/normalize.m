function [mu range Xnorm] = normalize(X)
 mu = mean(X);
 range = max(X) - min(X);
 Xnorm = (X - mu) ./ repmat(range, size(X, 1), 1);
end

