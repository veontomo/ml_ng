function Xnorm = normalize2(X, mu, range)
  Xnorm = (X - mu) ./ repmat(range, size(X, 1), 1);
end