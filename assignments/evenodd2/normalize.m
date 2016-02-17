function [mu scale Xnorm] = normalize(X)
 mu = mean(X);
 scale = max(X) - min(X);
 Xnorm = (X - mu)/scale;
end