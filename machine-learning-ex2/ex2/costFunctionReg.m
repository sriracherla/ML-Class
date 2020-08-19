function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

working_theta = theta;

h = sigmoid(X * working_theta);
plainJ = ((-y' * log(h)) - ((1-y)' * log(1 - h)))/ m;

J = plainJ + ((lambda/(2 * m)) * sum(theta(2:end,:).^2));
% in above note the theta(2:end, :) - this ignores theta zero - which we should
% also I split the compute into two for readability and split execution

% gradient compute
% calculate grad like before
temp_grad = ((1/m) * (X' * (h - y)));

% theta zero does not have lambda regulatization, so save this
grad_zero = temp_grad(1);

% simul. update all thetas (incl. zero) - we will reset zero below.
grad = temp_grad + (lambda/m) * theta;

% reset theta zero
grad(1) = grad_zero;





% =============================================================

end
