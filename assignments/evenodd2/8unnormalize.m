function X = unnormalize(mu, range, Xnorm)
 mu = mean(X);
 range = max(X) - min(X);
 Xnorm = (X - mu) ./ repmat(range, size(X, 1), 1);
end

