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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1a - Cost Function

% apply -1/m to
%     sum of
%         for each example i in 1:m
%             for each y-class k in 1:K
%                 compute cost

% Add ones to the X data matrix
X = [ones(m, 1) X]; 
% SIZE OF X = m by input_layer_size + 1

y_matrix = eye(num_labels)(y,:);
% SIZE OF Y_MATRIX = m by K (classes)

z2 = X * Theta1'; % SIZE = m by hidden_layer_size
a2 = sigmoid(z2); % SIZE = m by hidden_layer_size

n = size(a2, 1);
a2 = [ones(n, 1), a2]; % SIZE = m by hidden_layer_size + 1

z3 = a2 * Theta2'; % SIZE = m by K (classes)
a3 = sigmoid(z3);  % SIZE = m by K (classes) 
% This returns h for each example x in X for each class vector k in K

% COMMENTING EVERYTHING BELOW AS IT IS OPTIMIZED TO ONE LINE 

% temp_j = 0;
% for k = 1:num_labels
%     temp_h = a3(:, k);          % SIZE = m by 1
%     % size(temp_h)
%     temp_y = y_matrix(:, k);    % SIZE = m by 1
%     % size(temp_y)
%     temp_j = ((-temp_y' * log(temp_h)) - ((1-temp_y)' * log(1 - temp_h)))/m;
%   % plainJ = ((-y' * log(h)) - ((1-y)' * log(1 - h)))/ m;
%     J += sum(temp_j);
%     temp_j
% endfor

% Unpacking below
% trace - will return sum of the diagonal
% the rest is the same as Logisitic Regression Cost Function
J = trace(((-y_matrix' * log(a3)) - ((1 - y_matrix)' * log(1 - a3)))/ m);

% Part 1b - Lambda - Regularized Cost Function
reg_offset_theta1 = sum(sum(Theta1(:, 2: end) .^ 2)) * (lambda/ (2 * m));
reg_offset_theta2 = sum(sum(Theta2(:, 2: end) .^ 2)) * (lambda/ (2 * m));
reg_offset = (reg_offset_theta1 + reg_offset_theta2);

J = J + reg_offset;
% -------------------------------------------------------------

% Part 2- Backpropagation
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

% Step 1
% a1 or X is given
% calculate z2, a2, z3, a3 - done above
% note that I added bias units to X and a2 above
% note that y_matrix is also populated above {0, 1}

% Step 2
d3 = a3 - y_matrix;

% Step 3
% m = the number of training examples
% n = the number of training features, including the initial bias unit.
% h = the number of units in the hidden layer - NOT including the bias unit
% r = the number of output classifications

% d2 is tricky. It uses the (:,2:end) columns of Theta2. 
% d2 is the product of d3 and Theta2 (without the first column), 
% then multiplied element-wise by the sigmoid gradient of z2. 
% The size is (m x r) \cdot⋅ (r x h) --> (m x h). The size is the same as z2.

% Note: Excluding the first column of Theta2 is 
% because the hidden layer bias unit has no connection to 
% the input layer - so we do not use backpropagation for it. 
% See Figure 3 in ex4.pdf for a diagram showing this.

d2 = (d3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);

% Step 4
% Remove d2 (0) - done above

% Step 5
% Delta1 is the product of d2 and a1. The size is (h x m) \cdot⋅ (m x n) --> (h x n)
% Delta2 is the product of d3 and a2. The size is (r x m) \cdot⋅ (m x [h+1]) --> (r x [h+1])
Delta1 = d2' * X;
Delta2 = d3' * a2;

% Part 2b - Regularized Gradient
offset_gradient_layer1 = (lambda/ m) * Theta1;
offset_gradient_layer2 = (lambda/ m) * Theta2;

offset_gradient_layer1(:,1) = 0;
offset_gradient_layer2(:,1) = 0;

Theta1_grad = (Delta1/ m) + offset_gradient_layer1;
Theta2_grad = (Delta2/ m) + offset_gradient_layer2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
