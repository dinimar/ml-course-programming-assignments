function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% Add ones to the X data matrix
a1 = [ones(m, 1) X];
z2 = a1*Theta1';
a2 = sigmoid(z2);
% Add ones to the a2 data matrix
a2 = [ones(size(a2, 1), 1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);
h = a3;

% Create Y samples
Y = zeros(m, num_labels);
for i=1:m
    Y(i,y(i)) = 1;
end

% Regularization parameter
reg_j = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2))+sum(sum(Theta2(:, 2:end).^2)));

% Cost function
J = (1/m)*sum(sum((-Y).*log(h)-(1-Y).*log(1-h))) + reg_j;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
delta3 = [];
delta2 = [];

for t=1:m
%     Step 1
    y_t = Y(t, :);
    a1_t = X(t, :)';
    % Add ones to the a1_t data matrix
    a1_t = [1; a1_t];
    z2_t = Theta1*a1_t;
    a2_t = sigmoid(z2_t);
    % Add ones to the a2_t data matrix
    a2_t = [1; a2_t];
    z3_t = Theta2*a2_t;
    a3_t = sigmoid(z3_t);
%     Step 2
    delta3_t = a3_t-y_t';
%     Step 3
    z2_t = [1; z2_t];
    g_prime_t = sigmoidGradient(z2_t);
    delta2_t = (Theta2'*delta3_t).*g_prime_t;
%     Step 4
    delta2_t = delta2_t(2:end);
%     Step 5 
    Theta2_grad = (Theta2_grad + delta3_t*a2_t'); 
    Theta1_grad = (Theta1_grad + delta2_t*a1_t');
end

Theta2_grad = Theta2_grad ./ m;
Theta1_grad = Theta1_grad ./ m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:,2:end) = Theta1_grad(:,2:end)+(lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end)+(lambda/m)*Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
