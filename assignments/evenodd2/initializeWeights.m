%% generate random weights for a network with a given architecute
%% Here, we follow an advice on effective choice of initial parameters
%% and choose then from the interval (-eps_s, eps_s), where eps depends on the
%% number of units in layers s and s+1: eps_s = sqrt(6/(L_s + L_{s+1})).
%% Returns:
%% weights - unfolded version of the weight parameters
%% 
function weights = generateWeights(archit)
  requiredWeightNum = archit(2:end) * (1 + archit(1:end-1))';
  %% inner parts btw the layers
  inner = size(archit, 2) - 1;
  weights = [];
  for i = 1:inner
    epsilon = sqrt(6/(archit(i) + archit(i+1)));
    matr = epsilon*(2*rand(archit(i+1), archit(i)+1)-1)(:)';
    weights = [weights, matr];
  end

end