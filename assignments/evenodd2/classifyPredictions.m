%%% Finds true positive (tp), true negative (tn), false positivie (fp),
%%% false negative (fn) of the predictions Ypredicted w.r.t. actual values.

%%%             | actual    |  actual
%%%             | y = 1     |  y = 0 
%%%  ___________|___________|___________
%%%   predicted | true      | false 
%%%     y = 1   | positive  | positive
%%%  ___________|___________|__________
%%%   predicted | false     | true
%%%     y = 0   | negative  | negative
%%% ____________|_______________________

function [tp tn fp fn] = classifyPredictions(Yactual, Ypredicted)
  tp = sum(Yactual == 1 & Ypredicted == 1);
  fp = sum(Yactual == 0 & Ypredicted == 1);
  fn = sum(Yactual == 1 & Ypredicted == 0);
  tn = sum(Yactual == 0 & Ypredicted == 0);
end