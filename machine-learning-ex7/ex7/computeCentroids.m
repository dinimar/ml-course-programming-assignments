function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
% 
% C1 = [0, 0];
% C2 = [0, 0];
% C3 = [0, 0];
% K1 = 0;
% K2 = 0;
% K3 = 0;
% 
% for i=1:m
%     if idx(i) == 1
%         C1(1,1) = C1(1,1) + X(i,1);      
%         C1(1,2) = C1(1,2) + X(i,2);
%         K1 = K1 + 1;
%     else
%         if idx(i) == 2
%             C2(1,1) = C2(1,1) + X(i,1);
%             C2(1,2) = C2(1,2) + X(i,2);
%             K2 = K2 + 1;
%         else
%             if idx(i) == 3
%                 C3(1,1) = C3(1,1) + X(i,1);              
%                 C3(1,2) = C3(1,2) + X(i,2);
%                 K3 = K3 + 1;
%             end
%         end
%     end
% end
% 
% centroids(1,1) = C1(1,1) / K1;
% centroids(1,2) = C1(1,2) / K1;
% centroids(2,1) = C2(1,1) / K2;
% centroids(2,2) = C2(1,2) / K2;
% centroids(3,1) = C3(1,1) / K3;
% centroids(3,2) = C3(1,2) / K3;

for i = 1:K
    c_i = idx==i;
    n_i = sum(c_i);
    c_i_matrix = repmat(c_i,1,n);
    X_c_i = X .* c_i_matrix;
    centroids(i,:) = sum(X_c_i) ./ n_i;
end



% =============================================================


end

