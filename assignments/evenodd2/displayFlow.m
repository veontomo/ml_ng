function displayFlow(Jtraining, Jtest, Fscore)
  trainingSize = size(Jtraining, 2);
  subplot (2, 1, 1)
  hold on;
  plot(1:trainingSize, Jtraining, 'color', 'r')
  plot(1:trainingSize, Jtest, 'color', 'k')
  xlabel("training set size");
  ylabel("J min");
  hold off;
  subplot (2, 1, 2)
  plot(1:trainingSize, Fscore, 'color', 'b')
  xlabel("training set size");
  ylabel("F1");
end