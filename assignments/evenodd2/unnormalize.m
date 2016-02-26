function X = unnormalize(mu, range, Xnorm)
  X = Xnorm .* range + mu;
end

