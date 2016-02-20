function [J theta] = displayFlow(X, Y, trainingSize, testSize, lambda)
  Xtraining = X(1:trainingSize, :);
  Ytraining = Y(1:trainingSize);
  Xtest = X((trainingSize+1):(trainingSize + testSize), :);
  Ytest = Y((trainingSize+1):(trainingSize + testSize));
  Jtraining = zeros(1, trainingSize);
  Jtest = zeros(1, trainingSize);
  Fscore = zeros(1, trainingSize);
  options = optimset('GradObj', 'on', 'MaxIter', 400);
  theta_init = unifrnd(0, 1, 1, size(Xtraining, 2));
  for i = 1:trainingSize
    [theta, J, exit_flag] = fminunc(@(Theta)(cost(Xtraining(1:i, :), Ytraining(1:i, :), Theta, lambda)), theta_init, options);
    J2 = cost(Xtest, Ytest, theta, 0);
    Jtest(i) = J2;
    Jtraining(i) = J;
    %% Method precision, recall and accuracy
    Ypredicted = Xtest * theta' > 0;
    [tp tn fp fn] = classifyPredictions(Ytest, Ypredicted);
    Prec = tp/(tp + fp);
    Rec = tp/(tp + fn);
    % Acc = (tp + tn)/(tp + tn + fn + fp);
    Fscore(i) = 2*Prec*Rec/(Prec + Rec);
  endfor
  subplot (2, 1, 1)
  hold on;
  plot(1:trainingSize, Jtraining, 'color', 'r')
  plot(1:trainingSize, Jtest, 'color', 'k')
  hold off;
  subplot (2, 1, 2)
  plot(1:trainingSize, Fscore, 'color', 'b')
end