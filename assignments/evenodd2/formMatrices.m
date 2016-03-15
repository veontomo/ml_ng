%% Rearranges given row vector in a cell array of matrices corresponding to a
%% network with given architecture. 
%% 
%% weights - row vector
%% architecure - row vector defining the network architecture: each element of
%%               this row is the number of units in corresp. layer
%% orientation - "v" or "h" (vertical or horizontal). Specifies order in which 
%%               the row vector elements are inserted into matrices
%% Returns
%% c - cell array of size 1 x L, where L = (#layers - 1).
%%     If c = [a1 a2 a3 ...], then 
%%     c{1, 1} is a a2x(a1+1) matrix
%%     c{1, 2} is a a3x(a2+1) matrix
%%     ...
%%     c{1, L-1} is a aLx(a(L-1) + 1) matrix
%% s - column vector containing the total number of elements in the above cell
%%     array
function [c s] = formMatrices(weights, architecture, orientation)
  %% calculates the number of all weights that a neural network with a given
  %% architecture should have
  requiredWeightNum = architecture(2:end) * (1 + architecture(1:end-1))';
  weightsSize = size(weights, 2);
  if (requiredWeightNum != weightsSize) 
    printf("Warning: the network should have %u weights, while %u are given.\n", requiredWeightNum, weightsSize);
  endif;
  layersNum = size(architecture, 2);
  c = cell(1, layersNum - 1);
  s = zeros(1, layersNum - 1);
  %% # units in the previous layer
  prevLayerSize = architecture(1);
  %% # weights that have already been taken into consideration
  counter = 0;
  for layer = 2:layersNum
    layerSize = architecture(layer);
    length = (prevLayerSize + 1) * layerSize;
    s(layer - 1) = length;
    if orientation == "h"
      c(1, layer - 1) = reshape(weights((counter+1):(counter + length)), prevLayerSize + 1, layerSize)';
    else 
      c(1, layer - 1) = reshape(weights((counter+1):(counter + length)), layerSize, prevLayerSize + 1);
    end;
    counter = counter + length;
    prevLayerSize = layerSize;
  endfor
end;