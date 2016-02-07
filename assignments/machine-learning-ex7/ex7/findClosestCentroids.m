function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
for i = 1:size(X, 1)
%  printf("X(%d) = ", i);
%  printf("%f ", X(i, :));
%  printf("\n");
%  printf("centroids(%d) = ", 1);
%  printf("%f ", centroids(1, :));
%  printf("\n");
  dif =  X(i, :) - centroids(1, :);
  minD = dif * dif';
%  printf("distance to the first centroid: %f\n", minD);
  idx(i) = 1;
  for j = 2:K
    dif =  X(i, :) - centroids(j, :);
    tmpD = dif * dif';
%   printf("centroids(%d) = ", j);
%   printf("%f ", centroids(j, :));
%   printf("\n");
%   printf("distance to the centroid %d: %f\n", j, tmpD);
%   printf("tmp = %d, current min = %d\n", tmpD, minD);
    if (tmpD <= minD)
 %     printf("centroid %d is closer", j);
      minD = tmpD;
      idx(i) = j;
    endif;
  endfor;
endfor;






% =============================================================

end

