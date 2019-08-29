function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
r = size(g)(1);
c = size(g)(2);

for i = 1:r
  for j = 1:c
    g(i, j) = 1/(1+exp((-1)*z(i, j)));
  end  
end  



% =============================================================

end
